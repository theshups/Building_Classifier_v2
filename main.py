"""
main.py — BuildingYOLO
========================
Trains BOTH models then serves the FastAPI dashboard.

Time budget on AMD Ryzen 5 7535HS (CPU-only):
  ResNet50V2  classification  15 epochs  ~22 min
  YOLOv8n     detection       30 epochs  ~90 min
  Total                                  ~112 min

Usage:
  python main.py                  full pipeline (download + train both + serve)
  python main.py --train-only     download + train both, no server
  python main.py --serve          serve existing models only
  python main.py --resnet-only    train ResNet only
  python main.py --yolo-only      train YOLO only
  python main.py --skip-mit       skip 2.4 GB MIT download
  python main.py --force-manual   use data/manual/ for 3 classes
  python main.py --no-finetune    skip ResNet Phase B
  python main.py --show-graphs    plot accuracy/loss after training
  python main.py --host H         server host  (default 0.0.0.0)
  python main.py --port P         server port  (default 8000)
"""

import argparse
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"]      = "utf-8"

from pathlib import Path
from logger    import get_logger
from exception import AppException

log = get_logger("main")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser(description="BuildingYOLO")
    p.add_argument("--train-only",    action="store_true")
    p.add_argument("--serve",         action="store_true")
    p.add_argument("--resnet-only",   action="store_true")
    p.add_argument("--yolo-only",     action="store_true")
    p.add_argument("--skip-mit",      action="store_true")
    p.add_argument("--no-finetune",   action="store_true")
    p.add_argument("--show-graphs",   action="store_true")
    p.add_argument("--host",          default="0.0.0.0")
    p.add_argument("--port",          default=8000, type=int)
    return p.parse_args()


# ── Step 1: Data ──────────────────────────────────────────────────────────
def ingest(args):
    from src.data_ingestion import DataIngestion
    return DataIngestion(skip_mit=args.skip_mit).run()


def prepare(raw: dict):
    from src.data_transformation import DataTransformation
    pipeline_labels = raw.get("pipeline_labels", {})
    dt = DataTransformation(
        images          = raw,
        pipeline_labels = pipeline_labels,
    )
    dt.save_class_map()
    return dt


# ── Step 2a: ResNet50V2 ───────────────────────────────────────────────────
def train_resnet(dt, args) -> float:
    train_ds, val_ds, test_ds, class_names = dt.build_clf_datasets(
        img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    from src.model_trainer import ModelTrainer
    aug     = dt.get_augmentation_layer()
    trainer = ModelTrainer(
        num_classes        = len(class_names),
        augmentation_layer = aug,
        class_names        = class_names,
        img_size           = IMG_SIZE,
    )
    model = trainer.run(train_ds, val_ds, fine_tune=not args.no_finetune)

    log.info("Evaluating ResNet on test set ...")
    _, acc = model.evaluate(test_ds, verbose=1)
    log.info(f"ResNet test accuracy: {acc * 100:.1f}%")

    if args.show_graphs:
        _plot_history()

    return acc


# ── Step 2b: YOLOv8n detection ────────────────────────────────────────────
def train_yolo(dt) -> float:
    yaml_path = dt.build_yolo_dataset()

    from src.yolo_trainer import YOLOTrainer
    trainer = YOLOTrainer(
        yaml_path = str(yaml_path / "data.yaml"),
        epochs    = 30,
        imgsz     = 416,
        batch     = 16,
    )
    _, map50 = trainer.train()
    return map50


# ── Step 3: Serve ─────────────────────────────────────────────────────────
def serve(host: str, port: int):
    log.info("=" * 60)
    log.info("  BuildingYOLO  -  Starting server")
    log.info(f"  UI        -> http://{host}:{port}")
    log.info(f"  Analytics -> http://{host}:{port}/analytics/stream")
    log.info(f"  Health    -> http://{host}:{port}/health")
    log.info("=" * 60)
    import uvicorn
    uvicorn.run("src.app:app", host=host, port=port,
                reload=False, log_level="error")


# ── Graphs ────────────────────────────────────────────────────────────────
def _plot_history():
    import pandas as pd
    import matplotlib.pyplot as plt
    csv = Path("models/training_history.csv")
    if not csv.exists():
        log.warning("No training history CSV found.")
        return
    df  = pd.read_csv(csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BuildingYOLO - ResNet50V2", fontweight="bold")
    ax1.plot(df["epoch"], df["accuracy"],     label="Train", lw=2)
    ax1.plot(df["epoch"], df["val_accuracy"], label="Val",   lw=2)
    ax1.set_title("Accuracy"); ax1.set_ylim(0, 1)
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(df["epoch"], df["loss"],     label="Train", lw=2)
    ax2.plot(df["epoch"], df["val_loss"], label="Val",   lw=2)
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/training_history.png", dpi=150)
    plt.show()
    log.info("Graph saved -> models/training_history.png")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    t0   = time.time()

    log.info("=" * 60)
    log.info("  BuildingYOLO")
    log.info("  Models: ResNet50V2 (classification) + YOLOv8n (detection)")
    log.info("  Classes: exterior_facade, office_interior,")
    log.info("           warehouse, pipelines")
    log.info("  Est time: ~112 min on AMD Ryzen 5 7535HS")
    log.info("=" * 60)

    try:
        # Serve-only mode
        if args.serve:
            serve(args.host, args.port)
            sys.exit(0)

        # Check Roboflow key (needed unless force-manual)
        if not args.force_manual:
            key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
            if not key:
                log.error("ROBOFLOW_API_KEY not set. Run setup_roboflow.bat")
                sys.exit(1)

        # Data ingestion
        raw = ingest(args)
        dt  = prepare(raw)

        results = {}

        # Train ResNet50V2
        if not args.yolo_only:
            log.info("")
            log.info("--- ResNet50V2 Classification (~22 min) ---")
            results["resnet"] = train_resnet(dt, args)

        # Train YOLOv8n detection
        if not args.resnet_only:
            log.info("")
            log.info("--- YOLOv8n Detection (~90 min) ---")
            results["yolo"] = train_yolo(dt)

        # Summary
        elapsed = (time.time() - t0) / 60
        log.info("=" * 60)
        log.info(f"  Training complete in {elapsed:.1f} min")
        for m, acc in results.items():
            target = "TARGET MET" if acc >= 0.65 else "below target"
            log.info(f"  {m:8s}: {acc * 100:.1f}%  [{target}]")
        log.info("=" * 60)
        log.info("  Saved files:")
        log.info("    models/building_classifier.keras")
        log.info("    models/building_classifier.h5")
        log.info("    models/tokenizer.json")
        log.info("    models/yolo/runs/detector/weights/best.pt")
        log.info("=" * 60)

        if not args.train_only:
            serve(args.host, args.port)

    except AppException as exc:
        log.error(f"Failed: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")
        sys.exit(0)
