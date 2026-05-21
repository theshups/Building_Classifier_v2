"""
src/yolo_trainer.py — BuildingYOLO
=====================================
YOLOv8n detection model with bounding boxes.

Hardware : AMD Ryzen 5 7535HS (CPU only)
Target   : ~90 minutes  (30 epochs × ~3 min/epoch)
mAP@50   : 60-72% expected

Saves:
  models/yolo/runs/detector/weights/best.pt
  models/yolo/runs/detector/weights/last.pt
  models/yolo/detector_info.json
"""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger    import get_logger
from exception import AppException

log = get_logger(__name__)

MODELS_DIR = Path("models")
YOLO_DIR   = MODELS_DIR / "yolo"
RUNS_DIR   = YOLO_DIR / "runs"
BEST_PT    = RUNS_DIR / "detector" / "weights" / "best.pt"
LAST_PT    = RUNS_DIR / "detector" / "weights" / "last.pt"
INFO_JSON  = YOLO_DIR / "detector_info.json"
YAML_PATH  = Path("data/yolo_dataset/data.yaml")

CLASSES = ["exterior_facade", "office_interior", "warehouse", "pipelines"]
COLORS  = {
    "exterior_facade": (59,  130, 246),   # blue
    "office_interior": (34,  197,  94),   # green
    "warehouse":       (245, 158,  11),   # amber
    "pipelines":       (168,  85, 247),   # purple
}
CONF    = 0.25
IOU     = 0.45


class YOLOTrainer:
    def __init__(self, yaml_path=None, epochs=30, imgsz=416, batch=16):
        self.yaml   = yaml_path or str(YAML_PATH)
        self.epochs = epochs
        self.imgsz  = imgsz
        self.batch  = batch

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self):
        try:
            log.info("=" * 60)
            log.info("  BuildingYOLO  -  YOLOv8n Detection")
            log.info("  Model  : YOLOv8n (3.2M params, nano)")
            log.info(f"  Input  : {self.imgsz}x{self.imgsz}")
            log.info(f"  Epochs : {self.epochs}")
            log.info(f"  Batch  : {self.batch}")
            log.info(f"  Est    : ~{self.epochs * 3} min on CPU")
            log.info("=" * 60)

            try:
                from ultralytics import YOLO
            except ImportError:
                raise AppException("ultralytics not installed. Run install.bat", sys)

            if not Path(self.yaml).exists():
                raise AppException(
                    f"data.yaml not found at {self.yaml}. "
                    "Run: python main.py --prepare-data", sys)

            YOLO_DIR.mkdir(parents=True, exist_ok=True)
            RUNS_DIR.mkdir(parents=True, exist_ok=True)

            log.info("Loading YOLOv8n pretrained weights ...")
            model = YOLO("yolov8n.pt")   # detection, NOT -cls

            log.info("Training ...")
            model.train(
                data=self.yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                name="detector",
                project=str(RUNS_DIR),
                patience=10,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3,
                augment=True,
                mosaic=0.8,
                mixup=0.1,
                copy_paste=0.2,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                fliplr=0.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                verbose=True,
                exist_ok=True,
                device="cpu",
            )

            log.info("Validating ...")
            metrics  = model.val(data=self.yaml)
            map50    = float(metrics.box.map50) if hasattr(metrics, "box") else 0.0
            map50_95 = float(metrics.box.map)   if hasattr(metrics, "box") else 0.0

            log.info(f"mAP@50    : {map50    * 100:.1f}%")
            log.info(f"mAP@50-95 : {map50_95 * 100:.1f}%")

            self._save_info(map50, map50_95)

            log.info("=" * 60)
            log.info(f"  Best model -> {BEST_PT}")
            log.info(f"  mAP@50     -> {map50 * 100:.1f}%")
            log.info("=" * 60)
            return model, map50

        except Exception as exc:
            raise AppException(exc, sys) from exc

    def _save_info(self, map50: float, map50_95: float):
        INFO_JSON.parent.mkdir(parents=True, exist_ok=True)
        INFO_JSON.write_text(json.dumps({
            "model":       "YOLOv8n",
            "task":        "detection",
            "params":      "3.2M",
            "classes":     CLASSES,
            "num_classes": len(CLASSES),
            "imgsz":       self.imgsz,
            "epochs":      self.epochs,
            "map50":       round(map50    * 100, 2),
            "map50_95":    round(map50_95 * 100, 2),
            "model_path":  str(BEST_PT),
        }, indent=2), encoding="utf-8")

    # ── Inference with bounding boxes ─────────────────────────────────────
    @staticmethod
    def detect(image_bytes: bytes) -> dict:
        """
        Run YOLOv8n detection on raw image bytes.
        Returns annotated image (base64 JPEG) + detection list.
        """
        from ultralytics import YOLO

        pt = BEST_PT if BEST_PT.exists() else (LAST_PT if LAST_PT.exists() else None)
        if pt is None:
            raise FileNotFoundError(
                "No YOLO model found. Train first: python main.py")

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        model   = YOLO(str(pt))
        results = model(pil_img, conf=CONF, iou=IOU, verbose=False)

        detections = []
        result     = results[0]

        if result.boxes is not None and len(result.boxes):
            boxes  = result.boxes.xyxy.cpu().numpy()
            confs  = result.boxes.conf.cpu().numpy()
            clsids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cid in zip(boxes, confs, clsids):
                cls_name = CLASSES[cid] if cid < len(CLASSES) else str(cid)
                detections.append({
                    "class":      cls_name,
                    "confidence": round(float(conf), 4),
                    "box":        [round(float(v), 1) for v in box],  # x1,y1,x2,y2
                })

        annotated = YOLOTrainer._draw_boxes(pil_img, detections)
        return {
            "detections":   detections,
            "count":        len(detections),
            "annotated_b64": annotated,
        }

    @staticmethod
    def _draw_boxes(img: Image.Image, detections: list) -> str:
        """Draw bounding boxes with class labels on PIL image, return base64."""
        draw = ImageDraw.Draw(img)
        W, H = img.size

        # Try to load a font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", max(14, H // 40))
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            cls   = det["class"]
            conf  = det["confidence"]
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(cls, (255, 255, 255))

            # Box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label background
            label  = f"{cls}  {conf * 100:.0f}%"
            try:
                tw, th = draw.textsize(label, font=font)
            except Exception:
                tw, th = len(label) * 7, 14

            lx1 = max(x1, 0)
            ly1 = max(y1 - th - 4, 0)
            draw.rectangle([lx1, ly1, lx1 + tw + 6, ly1 + th + 4],
                           fill=color)
            draw.text((lx1 + 3, ly1 + 2), label,
                      fill=(255, 255, 255), font=font)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
