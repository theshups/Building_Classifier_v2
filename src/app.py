"""
src/app.py — BuildingYOLO FastAPI server
==========================================
Serves:
  GET  /                     → dark web UI
  POST /predict              → returns bounding boxes + classification probs
  GET  /health               → model status
  GET  /analytics            → current stats snapshot
  GET  /analytics/stream     → SSE live stream (updates every 2 s)

Both models run on every prediction:
  YOLOv8n   → bounding boxes + class + confidence per box
  ResNet50V2 → softmax probabilities over all 4 classes
"""

import asyncio
import io
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from logger import get_logger

log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
TOKENIZER  = Path("models/tokenizer.json")
MODEL_KERAS= Path("models/building_classifier.keras")
MODEL_H5   = Path("models/building_classifier.h5")
YOLO_BEST  = Path("models/yolo/runs/detector/weights/best.pt")
YOLO_LAST  = Path("models/yolo/runs/detector/weights/last.pt")
STATIC_IDX = Path("static/index.html")
IMG_SIZE   = (224, 224)

CLASSES = ["exterior_facade", "office_interior", "warehouse", "pipelines"]
DISPLAY = {
    "exterior_facade": "Exterior Facade",
    "office_interior": "Office Interior",
    "warehouse":       "Warehouse",
    "pipelines":       "Pipelines",
}

# ── Live stats ────────────────────────────────────────────────────────────
_stats = {
    "total":        0,
    "class_counts": {c: 0 for c in CLASSES},
    "latencies":    deque(maxlen=100),
    "feed":         deque(maxlen=50),
    "yolo_loaded":  False,
    "resnet_loaded":False,
}

# ── Models ────────────────────────────────────────────────────────────────
_yolo   = None
_resnet = None
_cls    = []


def _load_all():
    global _yolo, _resnet, _cls

    # YOLO detection
    pt = YOLO_BEST if YOLO_BEST.exists() else (
         YOLO_LAST if YOLO_LAST.exists() else None)
    if pt:
        try:
            from ultralytics import YOLO
            _yolo = YOLO(str(pt))
            _stats["yolo_loaded"] = True
            log.info(f"YOLOv8n loaded from {pt}")
        except Exception as exc:
            log.warning(f"YOLO load failed: {exc}")

    # ResNet50V2 classification
    for mp in (MODEL_KERAS, MODEL_H5):
        if mp.exists() and TOKENIZER.exists():
            try:
                import os, tensorflow as tf
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                _resnet = tf.keras.models.load_model(str(mp))
                tok     = json.loads(TOKENIZER.read_text(encoding="utf-8"))
                _cls    = tok["class_names"]
                _stats["resnet_loaded"] = True
                log.info(f"ResNet50V2 loaded from {mp}. Classes: {_cls}")
                break
            except Exception as exc:
                log.warning(f"ResNet load failed: {exc}")

    if not _stats["yolo_loaded"] and not _stats["resnet_loaded"]:
        log.warning("No models loaded. Run: python main.py --train-only")


def _run_yolo(image_bytes: bytes) -> dict:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tmp = "/tmp/_yolo_input.jpg"
    pil.save(tmp, quality=95)
    results    = _yolo(tmp, conf=0.25, iou=0.45, verbose=False)
    result     = results[0]
    detections = []
    if result.boxes is not None and len(result.boxes):
        boxes  = result.boxes.xyxy.cpu().numpy()
        confs  = result.boxes.conf.cpu().numpy()
        clsids = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cid in zip(boxes, confs, clsids):
            cls = CLASSES[cid] if cid < len(CLASSES) else str(cid)
            detections.append({
                "class":       cls,
                "display":     DISPLAY.get(cls, cls),
                "confidence":  round(float(conf), 4),
                "box":         [round(float(v), 1) for v in box],
            })

    # Annotate image
    from src.yolo_trainer import YOLOTrainer
    annotated = YOLOTrainer._draw_boxes(pil.copy(), detections)
    return {"detections": detections, "annotated_b64": annotated}


def _run_resnet(image_bytes: bytes) -> dict:
    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr   = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    probs = _resnet.predict(arr, verbose=0)[0]
    top   = int(np.argmax(probs))
    return {
        "top_class":      _cls[top] if top < len(_cls) else "unknown",
        "top_display":    DISPLAY.get(_cls[top], _cls[top]) if top < len(_cls) else "unknown",
        "confidence":     round(float(probs[top]), 4),
        "probabilities": [
            {"class":   c,
             "display": DISPLAY.get(c, c),
             "prob":    round(float(probs[i]), 4)}
            for i, c in enumerate(_cls)
        ],
    }


def _record(result: dict, ms: float):
    top = (result.get("classification", {}) or {}).get("top_class", "")
    _stats["total"] += 1
    if top in _stats["class_counts"]:
        _stats["class_counts"][top] += 1
    _stats["latencies"].append(ms)
    _stats["feed"].appendleft({
        "class":   DISPLAY.get(top, top),
        "conf":    round((result.get("classification", {}) or {}).get("confidence", 0) * 100, 1),
        "boxes":   len((result.get("detection", {}) or {}).get("detections", [])),
        "ms":      round(ms, 1),
        "ts":      time.strftime("%H:%M:%S"),
    })


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="BuildingYOLO", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    _load_all()


@app.get("/", response_class=HTMLResponse)
async def root():
    if STATIC_IDX.exists():
        return HTMLResponse(STATIC_IDX.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BuildingYOLO</h1><p>UI not found.</p>")


@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "yolo_loaded":    _stats["yolo_loaded"],
        "resnet_loaded":  _stats["resnet_loaded"],
        "classes":        DISPLAY,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(415, f"Unsupported: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if not _stats["yolo_loaded"] and not _stats["resnet_loaded"]:
        raise HTTPException(503, "No models loaded. Train first.")

    t0     = time.time()
    result = {}

    # YOLO detection
    if _stats["yolo_loaded"]:
        try:
            result["detection"] = _run_yolo(data)
        except Exception as exc:
            result["detection"] = {"error": str(exc)}
            log.warning(f"YOLO inference error: {exc}")

    # ResNet classification
    if _stats["resnet_loaded"]:
        try:
            result["classification"] = _run_resnet(data)
        except Exception as exc:
            result["classification"] = {"error": str(exc)}
            log.warning(f"ResNet inference error: {exc}")

    ms = round((time.time() - t0) * 1000, 1)
    result["latency_ms"] = ms
    _record(result, ms)

    top = (result.get("classification", {}) or {}).get("top_display", "")
    det = len((result.get("detection", {}) or {}).get("detections", []))
    log.info(f"Predicted: {top}  |  {det} boxes  |  {ms}ms")
    return JSONResponse(result)


@app.get("/analytics")
async def analytics():
    lat = list(_stats["latencies"])
    return JSONResponse({
        "total":        _stats["total"],
        "class_counts": _stats["class_counts"],
        "avg_latency":  round(sum(lat) / len(lat), 1) if lat else 0,
        "feed":         list(_stats["feed"])[:10],
        "yolo_loaded":  _stats["yolo_loaded"],
        "resnet_loaded":_stats["resnet_loaded"],
    })


@app.get("/analytics/stream")
async def stream(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            lat = list(_stats["latencies"])
            yield "data: " + json.dumps({
                "total":        _stats["total"],
                "class_counts": _stats["class_counts"],
                "avg_latency":  round(sum(lat) / len(lat), 1) if lat else 0,
                "feed":         list(_stats["feed"])[:10],
            }) + "\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
