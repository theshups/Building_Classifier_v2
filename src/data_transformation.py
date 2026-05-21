"""
src/data_transformation.py — BuildingYOLO
==========================================
Two responsibilities:

1. build_clf_datasets()
   Creates tf.data pipelines for ResNet50V2 classification.
   Images → 224×224, normalised [0,1], cached, prefetched.

2. build_yolo_dataset()
   Converts raw images + labels into YOLOv8 detection format:
     data/yolo_dataset/
       train/images/  train/labels/
       val/images/    val/labels/
       test/images/   test/labels/
       data.yaml
   Strategy:
     exterior_facade, office_interior, warehouse
       → whole-image bbox  (class_id 0.5 0.5 1.0 1.0)
     pipelines
       → Roboflow's real tight bboxes (re-mapped to class id 3)
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger    import get_logger
from exception import AppException

log = get_logger(__name__)

CLASSES   = ["exterior_facade", "office_interior", "warehouse", "pipelines"]
CLASS_ID  = {c: i for i, c in enumerate(CLASSES)}
CLF_ROOT  = Path("data/clf_splits")
YOLO_ROOT = Path("data/yolo_dataset")

CLF_SPLITS  = {"train": 0.70, "val": 0.15, "test": 0.15}
YOLO_SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED        = 42
MAX_IMG     = 1000   # per class


class DataTransformation:
    def __init__(self, images: dict, pipeline_labels: dict):
        self.images          = images
        self.pipeline_labels = pipeline_labels

    # ── 1. Classification tf.data datasets ───────────────────────────────
    def build_clf_datasets(self, img_size=(224, 224), batch_size=32):
        try:
            import tensorflow as tf
            import tensorflow.data  # noqa: F401

            log.info("=" * 60)
            log.info("  Building classification tf.data datasets")
            log.info(f"  Size: {img_size} | Batch: {batch_size}")
            log.info("=" * 60)

            self._write_clf_splits()

            AUTOTUNE = -1

            def _norm(x, y):
                return tf.cast(x, tf.float32) / 255.0, y

            def _load(path, shuffle):
                ds = tf.keras.utils.image_dataset_from_directory(
                    str(path),
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=SEED,
                    label_mode="int",
                )
                return ds, ds.class_names

            tr_raw, cls = _load(CLF_ROOT / "train", shuffle=True)
            va_raw, _   = _load(CLF_ROOT / "val",   shuffle=False)
            te_raw, _   = _load(CLF_ROOT / "test",  shuffle=False)

            def _pipe(ds):
                return (ds.map(_norm, num_parallel_calls=AUTOTUNE)
                          .cache()
                          .prefetch(AUTOTUNE))

            log.info(f"Classes: {cls}")
            return _pipe(tr_raw), _pipe(va_raw), _pipe(te_raw), cls

        except Exception as exc:
            raise AppException(exc, sys) from exc

    def _write_clf_splits(self):
        """Write image symlinks (copies) into train/val/test folder structure."""
        random.seed(SEED)
        for split in CLF_SPLITS:
            for cls in CLASSES:
                (CLF_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

        for cls in CLASSES:
            if cls == "pipelines":
                imgs = list(self.images.get("pipelines", []))
            else:
                imgs = list(self.images.get(cls, []))
            if not imgs:
                log.warning(f"  No images for '{cls}' in clf split.")
                continue
            random.shuffle(imgs)
            imgs = imgs[:MAX_IMG]
            n    = len(imgs)
            n_tr = int(n * CLF_SPLITS["train"])
            n_va = int(n * CLF_SPLITS["val"])
            split_map = {
                "train": imgs[:n_tr],
                "val":   imgs[n_tr:n_tr + n_va],
                "test":  imgs[n_tr + n_va:],
            }
            for split, files in split_map.items():
                dst_dir = CLF_ROOT / split / cls
                for i, src in enumerate(files):
                    ext = Path(src).suffix.lower() or ".jpg"
                    dst = dst_dir / f"{cls}_{i:05d}{ext}"
                    if not dst.exists():
                        shutil.copy2(src, dst)
            log.info(f"  clf '{cls}': train={n_tr} val={n_va} test={n-n_tr-n_va}")

    def get_augmentation_layer(self):
        import tensorflow as tf
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.RandomBrightness(0.20),
        ], name="augmentation")

    # ── 2. YOLO detection dataset ─────────────────────────────────────────
    def build_yolo_dataset(self) -> Path:
        try:
            log.info("=" * 60)
            log.info("  Building YOLO detection dataset")
            log.info("=" * 60)

            random.seed(SEED)
            self._make_yolo_dirs()

            totals = {s: 0 for s in YOLO_SPLITS}

            for cls in CLASSES:
                if cls == "pipelines":
                    imgs = list(self.images.get("pipelines", []))
                else:
                    imgs = list(self.images.get(cls, []))

                if not imgs:
                    log.warning(f"  No images for YOLO '{cls}' — skipping.")
                    continue

                random.shuffle(imgs)
                imgs = imgs[:MAX_IMG]
                n    = len(imgs)
                n_tr = int(n * YOLO_SPLITS["train"])
                n_va = int(n * YOLO_SPLITS["val"])
                split_map = {
                    "train": imgs[:n_tr],
                    "val":   imgs[n_tr:n_tr + n_va],
                    "test":  imgs[n_tr + n_va:],
                }
                cnt = {s: 0 for s in YOLO_SPLITS}
                for split, files in split_map.items():
                    for img_path in files:
                        if self._write_yolo_item(img_path, cls, split):
                            cnt[split]  += 1
                            totals[split] += 1

                log.info(f"  YOLO '{cls}': "
                         f"train={cnt['train']} val={cnt['val']} test={cnt['test']}")

            self._write_yaml()
            log.info(f"YOLO dataset -> {YOLO_ROOT.absolute()}")
            log.info(f"  Total: train={totals['train']} "
                     f"val={totals['val']} test={totals['test']}")
            return YOLO_ROOT

        except Exception as exc:
            raise AppException(exc, sys) from exc

    def _make_yolo_dirs(self):
        for split in YOLO_SPLITS:
            (YOLO_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
            (YOLO_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    def _write_yolo_item(self, img_path: Path,
                          cls_name: str, split: str) -> bool:
        try:
            img_path = Path(img_path)
            ext      = img_path.suffix.lower() or ".jpg"
            stem     = f"{cls_name}_{img_path.stem}"
            dst_img  = YOLO_ROOT / split / "images" / (stem + ext)
            dst_lbl  = YOLO_ROOT / split / "labels" / (stem + ".txt")
            if dst_img.exists() and dst_lbl.exists():
                return True
            shutil.copy2(img_path, dst_img)
            if cls_name == "pipelines":
                self._write_pipeline_label(img_path, dst_lbl)
            else:
                cid = CLASS_ID[cls_name]
                dst_lbl.write_text(
                    f"{cid} 0.500000 0.500000 1.000000 1.000000\n",
                    encoding="utf-8")
            return True
        except Exception as exc:
            log.warning(f"  Skipped {img_path.name}: {exc}")
            return False

    def _write_pipeline_label(self, img_path: Path, dst_lbl: Path):
        stem    = img_path.stem
        lbl_src = self.pipeline_labels.get(stem)
        cid     = CLASS_ID["pipelines"]
        if lbl_src and Path(lbl_src).exists():
            lines = []
            for line in Path(lbl_src).read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = parts
                    lines.append(f"{cid} {cx} {cy} {w} {h}")
            if lines:
                dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return
        dst_lbl.write_text(
            f"{cid} 0.500000 0.500000 1.000000 1.000000\n",
            encoding="utf-8")

    def _write_yaml(self):
        content = (
            f"path: {YOLO_ROOT.absolute()}\n"
            f"train: train/images\n"
            f"val:   val/images\n"
            f"test:  test/images\n\n"
            f"nc: {len(CLASSES)}\n"
            f"names: {CLASSES}\n"
        )
        (YOLO_ROOT / "data.yaml").write_text(content, encoding="utf-8")
        log.info(f"data.yaml written -> {YOLO_ROOT / 'data.yaml'}")

    def save_class_map(self):
        Path("models").mkdir(exist_ok=True)
        Path("models/tokenizer.json").write_text(
            json.dumps({
                "class_names":    CLASSES,
                "class_to_index": CLASS_ID,
                "num_classes":    len(CLASSES),
                "input_size":     [224, 224],
                "backbone":       "ResNet50V2",
            }, indent=2), encoding="utf-8")
        log.info("models/tokenizer.json saved.")
