"""
src/model_trainer.py — BuildingYOLO
=====================================
ResNet50V2 transfer-learning classifier.

Hardware : AMD Ryzen 5 7535HS (CPU only)
Target   : ~22 minutes  (15 epochs × ~1.5 min/epoch)
Accuracy : 65-75% (classification)

Saves:
  models/building_classifier.keras
  models/building_classifier.h5
  models/tokenizer.json
  models/training_history.csv
  models/checkpoints/best.keras
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
CKPT_DIR       = MODELS_DIR / "checkpoints"
HISTORY_CSV    = MODELS_DIR / "training_history.csv"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
MODEL_KERAS    = MODELS_DIR / "building_classifier.keras"
MODEL_H5       = MODELS_DIR / "building_classifier.h5"

# ── Hyper-parameters (tuned for 1-2 hr CPU budget) ───────────────────────
EPOCHS_HEAD     = 15     # ~22 min on Ryzen 5
EPOCHS_FINETUNE = 5      # ~7  min — total ~29 min
INITIAL_LR      = 1e-3
FINETUNE_LR     = 1e-5
FINETUNE_LAYERS = 30
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, phase: str, total: int):
        super().__init__()
        self.phase = phase
        self.total = total
        self._t    = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()
        log.info(f"[{self.phase}] Epoch {epoch + 1}/{self.total} ...")

    def on_epoch_end(self, epoch, logs=None):
        d = logs or {}
        log.info(
            f"[{self.phase}] Epoch {epoch + 1}/{self.total} "
            f"| loss={d.get('loss', 0):.4f} acc={d.get('accuracy', 0):.4f} "
            f"| val_loss={d.get('val_loss', 0):.4f} "
            f"val_acc={d.get('val_accuracy', 0):.4f} "
            f"| {time.time() - self._t:.0f}s"
        )


class ModelTrainer:
    def __init__(self, num_classes: int, augmentation_layer,
                 class_names: list, img_size=IMG_SIZE):
        self.num_classes        = num_classes
        self.augmentation_layer = augmentation_layer
        self.class_names        = class_names
        self.img_size           = img_size
        self.model              = None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True,   exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────
    def run(self, train_ds, val_ds, fine_tune: bool = True):
        try:
            log.info("=" * 60)
            log.info("  BuildingYOLO  -  ResNet50V2 Classification")
            log.info(f"  Classes : {self.num_classes}")
            log.info(f"  Input   : {self.img_size}")
            log.info(f"  Budget  : ~{EPOCHS_HEAD * 90 // 60} min (Phase A) "
                     f"+ ~{EPOCHS_FINETUNE * 90 // 60} min (Phase B)")
            log.info("=" * 60)

            self.model = self._build()
            cw         = self._class_weights(train_ds)

            # Phase A — head only
            log.info("-" * 60)
            log.info(f"Phase A: head only  |  {EPOCHS_HEAD} epochs  "
                     f"|  LR={INITIAL_LR}")
            log.info("-" * 60)
            hist_a   = self._fit("HEAD", INITIAL_LR, EPOCHS_HEAD,
                                  train_ds, val_ds, cw)
            best_a   = max(hist_a.history["val_accuracy"])
            all_hist = dict(hist_a.history)
            log.info(f"Phase A done  |  best val_acc={best_a * 100:.1f}%")

            # Phase B — fine-tune top 30 layers
            if fine_tune:
                log.info("-" * 60)
                log.info(f"Phase B: fine-tune top {FINETUNE_LAYERS} layers  "
                         f"|  {EPOCHS_FINETUNE} epochs  |  LR={FINETUNE_LR}")
                log.info("-" * 60)
                self._unfreeze()
                hist_b = self._fit("FINE", FINETUNE_LR, EPOCHS_FINETUNE,
                                    train_ds, val_ds, cw)
                best_b = max(hist_b.history["val_accuracy"])
                log.info(f"Phase B done  |  best val_acc={best_b * 100:.1f}%")
                for k in hist_b.history:
                    all_hist[k] = all_hist.get(k, []) + hist_b.history[k]

            best = max(all_hist.get("val_accuracy", [0]))
            self._save(all_hist)

            log.info("=" * 60)
            log.info(f"  Best val accuracy : {best * 100:.1f}%")
            log.info(f"  keras  -> {MODEL_KERAS}")
            log.info(f"  h5     -> {MODEL_H5}")
            log.info("=" * 60)
            return self.model

        except Exception as exc:
            raise AppException(exc, sys) from exc

    # ── Model architecture ────────────────────────────────────────────────
    def _build(self):
        log.info("Building ResNet50V2 model ...")
        base = tf.keras.applications.ResNet50V2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        log.info(f"  {base.name}  layers={len(base.layers)}  frozen")

        inputs = tf.keras.Input(shape=(*self.img_size, 3), name="input_image")
        x   = self.augmentation_layer(inputs, training=True)
        x   = tf.keras.applications.resnet_v2.preprocess_input(x * 255.0)
        x   = base(x, training=False)
        x   = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x   = tf.keras.layers.Dense(256, name="head_dense")(x)
        x   = tf.keras.layers.BatchNormalization(name="head_bn")(x)
        x   = tf.keras.layers.Activation("relu")(x)
        x   = tf.keras.layers.Dropout(0.40, name="head_drop1")(x)
        x   = tf.keras.layers.Dense(64, activation="relu", name="head_dense2")(x)
        x   = tf.keras.layers.Dropout(0.30, name="head_drop2")(x)
        out = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="predictions"
        )(x)

        m = tf.keras.Model(inputs, out, name="buildingyolo_resnet50v2")
        t = sum(tf.size(w).numpy() for w in m.trainable_weights)
        f = sum(tf.size(w).numpy() for w in m.non_trainable_weights)
        log.info(f"  Trainable: {t:,}  Frozen: {f:,}  Total: {t + f:,}")
        return m

    # ── Training ──────────────────────────────────────────────────────────
    def _fit(self, phase, lr, epochs, train_ds, val_ds, cw):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=cw,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    str(CKPT_DIR / "epoch_{epoch:03d}.keras"),
                    save_freq="epoch", verbose=0),
                tf.keras.callbacks.ModelCheckpoint(
                    str(CKPT_DIR / "best.keras"),
                    monitor="val_accuracy",
                    save_best_only=True, verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=4,
                    restore_best_weights=True, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.4,
                    patience=2, min_lr=1e-9, verbose=1),
                EpochLogger(phase, epochs),
            ],
        )

    def _unfreeze(self):
        base = next(
            l for l in self.model.layers if isinstance(l, tf.keras.Model)
        )
        base.trainable = True
        for layer in base.layers[:-FINETUNE_LAYERS]:
            layer.trainable = False
        t = sum(tf.size(w).numpy() for w in self.model.trainable_weights)
        log.info(f"Unfroze top {FINETUNE_LAYERS} layers. Trainable: {t:,}")

    # ── Class weights ─────────────────────────────────────────────────────
    def _class_weights(self, train_ds) -> dict:
        labels = []
        for _, bl in train_ds:
            labels.extend(bl.numpy().tolist())
        labels  = np.array(labels)
        classes = np.unique(labels)
        n       = len(labels)
        w       = {int(c): n / (len(classes) * np.sum(labels == c))
                   for c in classes}
        log.info(f"Class weights: {w}")
        return w

    # ── Save ──────────────────────────────────────────────────────────────
    def _save(self, history: dict):
        self.model.save(str(MODEL_KERAS))
        log.info(f"Saved keras  -> {MODEL_KERAS}")

        self.model.save(str(MODEL_H5), save_format="h5")
        log.info(f"Saved h5     -> {MODEL_H5}")

        TOKENIZER_PATH.write_text(json.dumps({
            "class_names":    self.class_names,
            "class_to_index": {c: i for i, c in enumerate(self.class_names)},
            "num_classes":    self.num_classes,
            "input_size":     list(self.img_size),
            "backbone":       "ResNet50V2",
            "version":        "BuildingYOLO",
        }, indent=2), encoding="utf-8")
        log.info(f"Tokenizer    -> {TOKENIZER_PATH}")

        if history:
            keys = list(history.keys())
            with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch"] + keys)
                for i, row in enumerate(zip(*[history[k] for k in keys]), 1):
                    w.writerow([i] + list(row))
            log.info(f"History CSV  -> {HISTORY_CSV}")
