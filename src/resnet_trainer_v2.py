"""
src/resnet_trainer_v2.py — BuildingYOLO
========================================
Complete rewrite targeting 70-80% accuracy.

Root causes fixed vs v1:
  1. Stacked Dropout(0.4+0.3) killed neurons → stuck at loss=log(4)=1.38
  2. LR=1e-3 too aggressive → reduced to 3e-4
  3. ResNet50V2 poor transfer for small dataset → switched to EfficientNetB0
  4. No oversampling → office_interior underrepresented
  5. Augmentation during Phase A → disabled until Phase B

Architecture:
  EfficientNetB0 (ImageNet, frozen in Phase A)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dropout(0.25)
  → Dense(4, softmax)

Phase A : 30 epochs  LR=3e-4  frozen backbone  no augmentation
Phase B : 20 epochs  LR=1e-5  unfreeze top 80 layers  light augmentation

Expected accuracy: 70-80%
Time on CPU      : ~35 min (Phase A) + ~25 min (Phase B) = ~60 min
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger    import get_logger
from exception import AppException

log = get_logger(__name__)

MODELS_DIR     = Path("models")
CKPT_DIR       = MODELS_DIR / "checkpoints_v2"
HISTORY_CSV    = MODELS_DIR / "training_history_v2.csv"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
MODEL_KERAS    = MODELS_DIR / "building_classifier.keras"
MODEL_H5       = MODELS_DIR / "building_classifier.h5"

CLF_SPLITS     = Path("data/clf_splits")
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32

EPOCHS_A       = 30     # Phase A: frozen backbone
EPOCHS_B       = 20     # Phase B: fine-tune
LR_A           = 3e-4   # Lower than v1 (was 1e-3)
LR_B           = 1e-5   # Fine-tune LR
UNFREEZE_LAYERS = 80    # Top layers to unfreeze in Phase B

CLASSES = ["exterior_facade", "office_interior", "pipelines", "warehouse"]


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, phase, total):
        super().__init__()
        self.phase = phase
        self.total = total
        self._t    = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        log.info(f"[{self.phase}] Epoch {epoch+1}/{self.total} ...")

    def on_epoch_end(self, epoch, logs=None):
        d = logs or {}
        log.info(
            f"[{self.phase}] Epoch {epoch+1}/{self.total}"
            f" | loss={d.get('loss',0):.4f} acc={d.get('accuracy',0):.4f}"
            f" | val_loss={d.get('val_loss',0):.4f}"
            f" val_acc={d.get('val_accuracy',0):.4f}"
            f" | {time.time()-self._t:.0f}s"
        )


class ResNetTrainerV2:
    def __init__(self):
        self.model = None
        self.class_names = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        try:
            log.info("=" * 60)
            log.info("  BuildingYOLO  -  ResNet Trainer V2")
            log.info("  Backbone : EfficientNetB0")
            log.info("  Changes  : lower LR, less dropout,")
            log.info("             oversampling, 2-phase training")
            log.info("  Target   : 70-80% accuracy")
            log.info("=" * 60)

            if not CLF_SPLITS.exists():
                raise AppException(
                    "data/clf_splits not found. "
                    "Run: python main.py --resnet-only first to create splits.", sys)

            train_ds, val_ds, test_ds, class_names = self._load_datasets()
            self.class_names = class_names
            log.info(f"Classes: {class_names}")

            # Check class imbalance
            self._check_imbalance(train_ds)

            # Build model
            self.model = self._build()

            all_hist = {}

            # Phase A: frozen backbone, no augmentation, 30 epochs
            log.info("-" * 60)
            log.info(f"Phase A: frozen backbone | {EPOCHS_A} epochs | LR={LR_A}")
            log.info("  No augmentation - letting backbone features work cleanly")
            log.info("-" * 60)
            aug_train = self._augment_dataset(train_ds, augment=False)
            hist_a    = self._fit("PHASE-A", LR_A, EPOCHS_A, aug_train, val_ds)
            best_a    = max(hist_a.history["val_accuracy"])
            all_hist  = dict(hist_a.history)
            log.info(f"Phase A done | best val_acc={best_a*100:.1f}%")

            # Phase B: unfreeze top layers, light augmentation, 20 epochs
            log.info("-" * 60)
            log.info(f"Phase B: unfreeze top {UNFREEZE_LAYERS} layers"
                     f" | {EPOCHS_B} epochs | LR={LR_B}")
            log.info("-" * 60)
            self._unfreeze()
            aug_train_b = self._augment_dataset(train_ds, augment=True)
            hist_b      = self._fit("PHASE-B", LR_B, EPOCHS_B, aug_train_b, val_ds)
            best_b      = max(hist_b.history["val_accuracy"])
            log.info(f"Phase B done | best val_acc={best_b*100:.1f}%")
            for k in hist_b.history:
                all_hist[k] = all_hist.get(k, []) + hist_b.history[k]

            # Evaluate on test set
            log.info("Evaluating on test set ...")
            loss, acc = self.model.evaluate(test_ds, verbose=1)
            log.info(f"Test accuracy : {acc*100:.1f}%")
            log.info(f"Test loss     : {loss:.4f}")

            self._save(all_hist)

            best = max(all_hist.get("val_accuracy", [0]))
            log.info("=" * 60)
            log.info(f"  Best val accuracy : {best*100:.1f}%")
            log.info(f"  Test accuracy     : {acc*100:.1f}%")
            if acc >= 0.70:
                log.info("  TARGET 70% MET!")
            else:
                log.info(f"  {acc*100:.1f}% — run more epochs or add data")
            log.info("=" * 60)

            return self.model, acc

        except Exception as exc:
            raise AppException(exc, sys) from exc

    # ── Dataset loading with oversampling ────────────────────────────────
    def _load_datasets(self):
        AUTOTUNE = -1

        def _norm(x, y):
            return tf.cast(x, tf.float32), y   # keep [0,255] for EfficientNet

        def _load(path, shuffle):
            ds = tf.keras.utils.image_dataset_from_directory(
                str(path), image_size=IMG_SIZE,
                batch_size=None,   # unbatched for oversampling
                shuffle=shuffle, seed=42, label_mode="int")
            return ds, ds.class_names

        train_raw, cls = _load(CLF_SPLITS / "train", shuffle=True)
        val_raw,   _   = _load(CLF_SPLITS / "val",   shuffle=False)
        test_raw,  _   = _load(CLF_SPLITS / "test",  shuffle=False)

        # Oversample minority classes to balance training set
        train_balanced = self._oversample(train_raw, cls)

        # Batch and prefetch
        def _pipe(ds):
            return (ds.map(_norm, num_parallel_calls=AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .cache()
                      .prefetch(AUTOTUNE))

        return (_pipe(train_balanced),
                _pipe(val_raw.map(_norm, num_parallel_calls=AUTOTUNE)
                                .batch(BATCH_SIZE)),
                _pipe(test_raw.map(_norm, num_parallel_calls=AUTOTUNE)
                                .batch(BATCH_SIZE)),
                cls)

    def _oversample(self, ds, class_names):
        """Oversample minority classes so all classes have equal representation."""
        log.info("Checking class distribution ...")

        # Count per class
        counts = {i: 0 for i in range(len(class_names))}
        items  = list(ds)
        for img, lbl in items:
            counts[int(lbl)] += 1

        max_count = max(counts.values())
        for i, name in enumerate(class_names):
            log.info(f"  {name}: {counts[i]} images"
                     f" (will oversample to {max_count})")

        # Split by class
        class_ds = {i: [] for i in range(len(class_names))}
        for img, lbl in items:
            class_ds[int(lbl)].append((img, lbl))

        # Oversample each class to max_count
        balanced = []
        for cls_idx, samples in class_ds.items():
            if not samples:
                continue
            n_repeat = max_count // len(samples) + 1
            oversampled = (samples * n_repeat)[:max_count]
            balanced.extend(oversampled)

        np.random.seed(42)
        np.random.shuffle(balanced)

        imgs   = tf.data.Dataset.from_tensor_slices(
            [x[0] for x in balanced])
        labels = tf.data.Dataset.from_tensor_slices(
            [x[1] for x in balanced])

        log.info(f"  After oversampling: {len(balanced)} total images")
        return tf.data.Dataset.zip((imgs, labels))

    def _augment_dataset(self, ds, augment: bool):
        """Apply light augmentation only in Phase B."""
        if not augment:
            return ds

        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
        ], name="light_aug")

        AUTOTUNE = -1
        return ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=AUTOTUNE)

    # ── Model architecture ───────────────────────────────────────────────
    def _build(self):
        log.info("Building EfficientNetB0 model ...")
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        log.info(f"  {base.name}  layers={len(base.layers)}  frozen")

        inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

        # EfficientNetB0 expects [0, 255] — no manual preprocess needed
        x   = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x   = base(x, training=False)
        x   = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x   = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
        x   = tf.keras.layers.Dropout(0.25, name="head_drop")(x)
        out = tf.keras.layers.Dense(
            len(self.class_names), activation="softmax", name="predictions"
        )(x)

        m = tf.keras.Model(inputs, out, name="buildingyolo_efficientnetb0")
        t = sum(tf.size(w).numpy() for w in m.trainable_weights)
        f = sum(tf.size(w).numpy() for w in m.non_trainable_weights)
        log.info(f"  Trainable: {t:,}  Frozen: {f:,}  Total: {t+f:,}")
        return m

    # ── Training ─────────────────────────────────────────────────────────
    def _fit(self, phase, lr, epochs, train_ds, val_ds):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                label_smoothing=0.1),   # label smoothing helps generalisation
            metrics=["accuracy"],
        )
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    str(CKPT_DIR / "best.keras"),
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=8,
                    restore_best_weights=True,
                    verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-9,
                    verbose=1),
                EpochLogger(phase, epochs),
            ],
        )

    # ── Unfreeze top layers ──────────────────────────────────────────────
    def _unfreeze(self):
        base = next(
            l for l in self.model.layers
            if isinstance(l, tf.keras.Model))
        base.trainable = True
        for layer in base.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False
        t = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        log.info(f"Unfroze top {UNFREEZE_LAYERS} layers. Trainable: {t:,}")

    def _check_imbalance(self, train_ds):
        log.info("Class distribution check:")
        counts = {}
        for _, lbls in train_ds:
            for l in lbls.numpy():
                counts[int(l)] = counts.get(int(l), 0) + 1
        total = sum(counts.values())
        for idx, cnt in sorted(counts.items()):
            name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            log.info(f"  Class {idx} ({name}): {cnt} ({cnt/total*100:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    def _save(self, history: dict):
        self.model.save(str(MODEL_KERAS))
        log.info(f"Saved -> {MODEL_KERAS}")

        self.model.save(str(MODEL_H5), save_format="h5")
        log.info(f"Saved -> {MODEL_H5}")

        TOKENIZER_PATH.write_text(json.dumps({
            "class_names":    self.class_names,
            "class_to_index": {c: i for i, c in enumerate(self.class_names)},
            "num_classes":    len(self.class_names),
            "input_size":     list(IMG_SIZE),
            "backbone":       "EfficientNetB0",
            "preprocessing":  "efficientnet_preprocess_input",
            "version":        "BuildingYOLO-v2",
        }, indent=2), encoding="utf-8")
        log.info(f"Tokenizer -> {TOKENIZER_PATH}")

        if history:
            keys = list(history.keys())
            with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch"] + keys)
                for i, row in enumerate(zip(*[history[k] for k in keys]), 1):
                    w.writerow([i] + list(row))
            log.info(f"History   -> {HISTORY_CSV}")
