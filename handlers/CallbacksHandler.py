import os
import datetime
from typing import Optional, List, Dict

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .base_handler import BaseHandler
from managers import DirectoryManager


class LearningRateLogger(Callback):
    """Logs current learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'learning_rate'):
            lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        elif hasattr(optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(optimizer.lr))
        else:
            return

        print(f" - LR: {lr:.2e}")
        if logs is not None:
            logs['lr'] = lr


class PerClassF1Callback(Callback):
    """Tracks per-class F1-score on the validation set after each epoch."""

    def __init__(self, val_generator: ImageDataGenerator, class_labels: List[str]):
        super().__init__()
        self.val_generator   = val_generator
        self.class_labels    = class_labels
        self.epoch_class_f1: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch, logs=None):
        self.val_generator.reset()
        y_pred = np.argmax(self.model.predict(self.val_generator, verbose=0), axis=1)
        y_true = self.val_generator.classes

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0
        )
        self.epoch_class_f1.append({
            label: report[label]['f1-score']
            for label in self.class_labels
        })


class CallbacksHandler(BaseHandler):
    """Builds and manages Keras callbacks for FER training experiments."""

    def __init__(self, model_name: str, directory_manager: DirectoryManager):
        self.model_name               = model_name
        self.root_directory           = directory_manager.get('root')
        self.archive_directory        = directory_manager.get('archive')
        self.logs_directory           = directory_manager.get('logs')
        self.visualizations_directory = directory_manager.get('training_visualizations')

        self.model_path                = os.path.join(self.root_directory, f'{self.model_name}.keras')
        self.callbacks: List[Callback] = []
        self.per_class_f1_callback: Optional[PerClassF1Callback] = None

        super().__init__(self.visualizations_directory)
        print('CallbacksHandler has been initialized.')

    # ── private builders ────────────────────────────────────

    def _build_per_class_f1(self, val_generator: ImageDataGenerator, class_labels: List[str]) -> PerClassF1Callback:
        self.per_class_f1_callback = PerClassF1Callback(val_generator, class_labels)
        return self.per_class_f1_callback

    def _build_checkpoint(self) -> ModelCheckpoint:
        """Save best model weights based on val_loss."""
        return ModelCheckpoint(
            filepath=self.model_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

    def _build_early_stopping(self) -> Optional[EarlyStopping]:
        """Return EarlyStopping callback or None if disabled in CONFIG."""
        cfg = CONFIG['callbacks']['early_stopping']
        if not cfg['enabled']:
            return None
        return EarlyStopping(
            monitor='val_loss',
            patience=cfg['patience'],
            min_delta=cfg['min_delta'],
            restore_best_weights=True,
            verbose=1
        )

    def _build_reduce_lr(self) -> Optional[ReduceLROnPlateau]:
        """Return ReduceLROnPlateau callback or None if disabled in CONFIG."""
        cfg = CONFIG['callbacks']['reduce_lr']
        if not cfg['enabled']:
            return None
        return ReduceLROnPlateau(
            monitor='val_loss',
            factor=cfg['factor'],
            patience=cfg['patience'],
            min_delta=1e-4,
            min_lr=cfg['min_lr'],
            verbose=1
        )

    def _build_tensorboard(self) -> TensorBoard:
        """Return TensorBoard callback with timestamped log directory."""
        log_dir = os.path.join(
            self.logs_directory,
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        return TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )

    def _build_csv_logger(self) -> CSVLogger:
        """Return CSVLogger writing to the archive directory."""
        path = os.path.join(self.archive_directory, 'training.csv')
        return CSVLogger(path, append=False)

    # ── public ──────────────────────────────────────────────

    def create(self, val_generator, class_labels: List[str]) -> List[Callback]:
        """Build all callbacks from global CONFIG."""
        self.callbacks = []

        # Always included
        self.callbacks.append(self._build_checkpoint())
        self.callbacks.append(self._build_tensorboard())
        self.callbacks.append(self._build_csv_logger())
        self.callbacks.append(LearningRateLogger())

        if val_generator is not None and class_labels is not None:
            self.callbacks.append(self._build_per_class_f1(val_generator, class_labels))

        # Optional — included based on CONFIG
        for builder in (self._build_early_stopping, self._build_reduce_lr):
            cb = builder()
            if cb is not None:
                self.callbacks.append(cb)

        print('Callbacks have been created.')
        return self.callbacks

    # ── summary ──────────────────────────────────────────────

    def generate_summary(self, mode: str) -> None:
        es  = CONFIG['callbacks']['early_stopping']
        rlr = CONFIG['callbacks']['reduce_lr']

        checkpoint_path = os.path.join(self.root_directory, f'{self.model_name}.keras')
        csv_path        = os.path.join(self.archive_directory, 'training.csv')
        logs_path       = self.logs_directory

        summary_data = [
            ('ModelCheckpoint',    'on'),
            ('monitor',            'val_loss (min)'),
            ('path',               checkpoint_path),
            None,
            ('TensorBoard',        'on'),
            ('log_dir',            logs_path),
            None,
            ('CSVLogger',          'on'),
            ('path',               csv_path),
            None,
            ('LearningRateLogger', 'on'),
            None,
            ('EarlyStopping',      'on' if es['enabled'] else 'off'),
            ('patience',           es['patience']  if es['enabled'] else 'n/a'),
            ('min_delta',          es['min_delta'] if es['enabled'] else 'n/a'),
            None,
            ('ReduceLROnPlateau',  'on' if rlr['enabled'] else 'off'),
            ('patience',           rlr['patience'] if rlr['enabled'] else 'n/a'),
            ('factor',             rlr['factor']   if rlr['enabled'] else 'n/a'),
            ('min_lr',             rlr['min_lr']   if rlr['enabled'] else 'n/a'),
        ]

        if mode == 'latex':
            self._generate_latex_summary('CallbacksHandler', summary_data, 'callbacks_summary.tex')
        else:
            self._generate_ascii_summary('CallbacksHandler', summary_data)