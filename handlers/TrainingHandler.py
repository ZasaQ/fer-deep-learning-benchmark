import datetime
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .BaseHandler import BaseHandler
from .DataAugmentationHandler import DataAugmentationHandler
from ExperimentMetrics import TrainingMetricsMixin


class TrainingHandler(TrainingMetricsMixin, BaseHandler):
    """Handles model training execution and learning curve visualization."""

    def __init__(self,
                 config: dict,
                 model: tf.keras.Model,
                 callbacks: List[Callback],
                 data_augmentation_handler: DataAugmentationHandler,
                 visualizations_directory: str):
        self.config                    = config
        self.model                     = model
        self.callbacks                 = callbacks
        self.data_augmentation_handler = data_augmentation_handler
        super().__init__(visualizations_directory)

        self.history:     Optional[tf.keras.callbacks.History] = None
        self.fit_start:   Optional[datetime.datetime]          = None
        self.fit_stop:    Optional[datetime.datetime]          = None
        self.fit_elapsed: Optional[float]                      = None
        self.device_info: dict                                 = self._collect_device_info()

        self.best_val_accuracy: Optional[float] = None
        self.best_val_loss:     Optional[float] = None
        self.best_epoch:        Optional[int]   = None
        self.epochs_run:        Optional[int]   = None

        self.best_val_acc_epoch: Optional[int] = None

        self.acc_gap:  Optional[np.ndarray] = None
        self.loss_gap: Optional[np.ndarray] = None

        print('TrainingHandler has been initialized.')

    # ── private helpers ──────────────────────────────────────

    @staticmethod
    def _collect_device_info() -> dict:
        """Collect GPU info via TensorFlow + nvidia-smi."""
        info = {
            'device':    'CPU',
            'gpu_count': 0,
            'gpus':      [],
        }

        try:
            gpus = tf.config.list_physical_devices('GPU')
            info['gpu_count'] = len(gpus)

            if not gpus:
                print("Device: CPU (no GPU detected)")
                return info

            info['device'] = 'GPU'

            for gpu in gpus:
                gpu_entry = {'name': gpu.name}

                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    gpu_entry['device_name']        = details.get('device_name', 'unknown')
                    gpu_entry['compute_capability'] = details.get('compute_capability', 'unknown')
                except Exception:
                    gpu_entry['device_name']        = 'unknown'
                    gpu_entry['compute_capability'] = 'unknown'

                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi',
                         '--query-gpu=name,memory.total,memory.free',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        idx   = gpus.index(gpu)
                        if idx < len(lines):
                            parts = [p.strip() for p in lines[idx].split(',')]
                            if len(parts) == 3:
                                gpu_entry['smi_name']        = parts[0]
                                gpu_entry['memory_total_mb'] = int(parts[1])
                                gpu_entry['memory_free_mb']  = int(parts[2])
                except Exception:
                    pass

                info['gpus'].append(gpu_entry)

            for g in info['gpus']:
                name      = g.get('smi_name', g.get('device_name', 'GPU'))
                mem_total = g.get('memory_total_mb')
                mem_free  = g.get('memory_free_mb')
                mem_str   = f"  {mem_total} MB total / {mem_free} MB free" if mem_total else ''
                print(f"Device: {name}{mem_str}")

        except Exception as e:
            info['error'] = str(e)
            print(f"Device info collection failed: {e}")

        return info

    def _build_class_weights(self) -> Optional[dict]:
        class_weights = self.config['class_weights']
        if not class_weights.get('enabled', False):
            return None

        if class_weights.get('mode') != 'balanced':
            print("Class weights mode 'manual' not implemented – skipping.")
            return None

        generator = self.data_augmentation_handler.train_generator
        classes   = generator.classes

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(classes),
            y=classes
        )
        class_weight_dict = dict(enumerate(weights))

        print("Class weights (balanced):")
        for idx, w in class_weight_dict.items():
            print(f"  class {idx}: {w:.4f}")

        return class_weight_dict

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format seconds as Xh Ym Zs."""
        seconds = int(seconds)
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    # ── training ─────────────────────────────────────────────

    def train(self) -> tf.keras.callbacks.History:
        """Run model.fit with generators and CONFIG parameters."""
        if self.data_augmentation_handler.train_generator is None:
            raise RuntimeError(
                "Generators not created. "
                "Call data_augmentation_handler.create_generators() first."
            )

        self.fit_start = datetime.datetime.now()
        print(f"Training started: {self.fit_start.strftime('%Y-%m-%d %H:%M:%S')}")

        self.history = self.model.fit(
            self.data_augmentation_handler.train_generator,
            validation_data=self.data_augmentation_handler.val_generator,
            epochs=self.config['epochs'],
            callbacks=self.callbacks,
            class_weight=self._build_class_weights(),
            verbose=1
        )

        self.fit_stop    = datetime.datetime.now()
        self.fit_elapsed = (self.fit_stop - self.fit_start).total_seconds()

        self.train_acc = self.history.history.get('accuracy', [])
        self.val_acc   = self.history.history.get('val_accuracy', [])
        self.train_loss = self.history.history.get('loss', [])
        self.val_loss  = self.history.history.get('val_loss', [])

        self.best_val_acc_epoch = int(np.argmax(self.history.history['val_accuracy'])) + 1

        self.best_epoch        = int(np.argmin(self.val_loss)) + 1
        self.best_val_loss     = float(min(self.val_loss))
        self.best_val_accuracy = float(max(self.val_acc))
        self.epochs_run        = len(self.val_loss)
        
        if self.train_acc and self.val_acc:
            self.acc_gap = np.array(self.train_acc) - np.array(self.val_acc)
            
        if self.train_loss and self.val_loss:
            self.loss_gap = np.array(self.val_loss) - np.array(self.train_loss)

        self.early_stopped = self.epochs_run < self.config['epochs']
        self.train_val_gap = round(float(self.acc_gap[-1]), 4) if self.acc_gap is not None else None

        print(f"\nTraining complete.")
        print(f"  Started:           {self.fit_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Stopped:           {self.fit_stop.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Elapsed:           {self._fmt_duration(self.fit_elapsed)}")
        print(f"  Epochs:            {self.epochs_run}/{self.config['epochs']}"
              + (" (early stopping)" if self.early_stopped else ""))
        print(f"  Best epoch:        {self.best_epoch}")
        print(f"  Best val_loss:     {self.best_val_loss:.4f}")
        print(f"  Best val_accuracy: {self.best_val_accuracy:.4f}")
        print(f"  Train/val gap:     {self.train_val_gap}" if self.train_val_gap is not None else "  Train/val gap:     n/a")

        return self.history

    @staticmethod
    def _smart_annotate(ax, x, y, label, color, x_range, offset_right=(8, -14), offset_left=(-8, -14)):
        """Annotate a point, flipping to the left if too close to the right edge."""
        threshold = 0.90 * x_range
        if x > threshold:
            xytext, ha = offset_left, 'right'
        else:
            xytext, ha = offset_right, 'left'
        ax.annotate(
            label,
            xy=(x, y),
            xytext=xytext,
            textcoords='offset points',
            fontsize=8, fontweight='bold', color=color, ha=ha
        )

    # ── visualizations ───────────────────────────────────────

    def plot_learning_curves(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        """Plot training and validation accuracy and loss over epochs."""
        if not self._guard(self.history is not None, "No training history. Call train() first."):
            return

        history      = self.history.history
        epochs = range(1, len(history['loss']) + 1)

        COLOR_TRAIN = '#2980b9'
        COLOR_VAL   = '#e74c3c'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Accuracy PLot
        final_train_acc = history['accuracy'][-1]
        final_val_acc   = history['val_accuracy'][-1]

        ax1.plot(epochs, history['accuracy'],     label='Train', linewidth=2, color=COLOR_TRAIN)
        ax1.plot(epochs, history['val_accuracy'], label='Val',   linewidth=2, color=COLOR_VAL)
        ax1.axvline(x=self.best_val_acc_epoch, color=COLOR_VAL, linestyle=':', label='Max val accuracy')
        ax1.set_xlim(1, max(epochs))
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(
            f'Training & Validation Accuracy\n'
            f'Final Train: {final_train_acc:.4f}  |  Final Val: {final_val_acc:.4f}'
        )
        ticks = sorted(set([int(t) for t in ax1.get_xticks() if 1 <= t <= max(epochs)] + [self.best_val_acc_epoch]))
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([f'*{t}' if t == self.best_val_acc_epoch else str(t) for t in ticks])
        self._smart_annotate(
            ax1, self.best_val_acc_epoch, history['val_accuracy'][self.best_val_acc_epoch - 1],
            f'max: {history["val_accuracy"][self.best_val_acc_epoch-1]:.4f}',
            COLOR_VAL, x_range=max(epochs),
            offset_right=(8, -14), offset_left=(-8, -14)
        )
        ax1.legend()
        ax1.grid(axis='both', alpha=0.3)

        # Loss Plot
        final_train_loss = history['loss'][-1]
        final_val_loss   = history['val_loss'][-1]

        ax2.plot(epochs, history['loss'],     label='Train', linewidth=2, color=COLOR_TRAIN)
        ax2.plot(epochs, history['val_loss'], label='Val',   linewidth=2, color=COLOR_VAL)
        ax2.axvline(x=self.best_epoch, color=COLOR_VAL, linestyle=':', label='Min val loss')
        ax2.set_xlim(1, max(epochs))
        ax2.set_ylim(0, max(max(history['loss']), max(history['val_loss'])) * 1.1)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(
            'Training & Validation Loss\n'
            f'Final Train: {final_train_loss:.4f}  |  Final Val: {final_val_loss:.4f}'
        )
        ticks = sorted(set([int(t) for t in ax2.get_xticks() if 1 <= t <= max(epochs)] + [self.best_epoch]))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f'*{t}' if t == self.best_epoch else str(t) for t in ticks])
        self._smart_annotate(
            ax2, self.best_epoch, history['val_loss'][self.best_epoch - 1],
            f'min: {history["val_loss"][self.best_epoch-1]:.4f}',
            COLOR_VAL, x_range=max(epochs),
            offset_right=(8, 8), offset_left=(-8, 8)
        )
        ax2.legend()
        ax2.grid(axis='both', alpha=0.3)

        plt.suptitle("Learning Curves")
        self._save_fig('learning_curves.png')

    def plot_learning_rate_schedule(self, figsize: Tuple[int, int] = (10, 5)) -> None:
        """Plot learning rate value over epochs on a log scale."""
        if not self._guard(self.history is not None,
                       "No training history. Call train() first."):
            return

        if 'lr' not in self.history.history:
            print("No learning rate data in history.")
            return

        h      = self.history.history
        epochs = range(1, len(h['lr']) + 1)
        lr_max = max(h['lr'])

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(epochs, h['lr'], linewidth=2, color='#27ae60')
        ax.fill_between(epochs, h['lr'], alpha=0.1, color='#27ae60')
        ax.annotate(f'start: {lr_max:.2e}',
                    xy=(1, lr_max),
                    xytext=(8, 4), textcoords='offset points', fontsize=8, color='gray')
        ax.annotate(f'end: {h["lr"][-1]:.2e}',
                    xy=(len(epochs), h['lr'][-1]),
                    xytext=(-60, 4), textcoords='offset points', fontsize=8, color='gray')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.set_xlim(1, max(epochs))
        ax.grid(axis='both', alpha=0.3)

        plt.suptitle("Learning Rate Schedule")
        self._save_fig('learning_rate_schedule.png')

    def plot_train_val_gap(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        """Plot the gap between training and validation metrics over epochs."""
        if not self._guard(self.history is not None, "No training history. Call train() first."):
            return

        if self.acc_gap is None or self.loss_gap is None:
            print("Gap metrics not found. Should exist in history.")
            return

        epochs     = range(1, self.epochs_run + 1)
        best_epoch = self.best_epoch

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        max_acc_gap_epoch  = int(np.argmax(np.abs(self.acc_gap))) + 1
        max_loss_gap_epoch = int(np.argmax(np.abs(self.loss_gap))) + 1

        # Accuracy gap
        ax1.plot(epochs, self.acc_gap, linewidth=2, color='#e74c3c', label='Train - Val Accuracy')
        ax1.fill_between(epochs, 0, self.acc_gap, alpha=0.15, color='#e74c3c')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        ax1.axhspan(-0.02, 0.05, color='green', alpha=0.06, label='Healthy zone')
        ax1.axvline(x=max_acc_gap_epoch, color='#e74c3c', linestyle=':', alpha=0.7, label='Max gap')
        ax1.set_xlim(1, max(epochs))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy Gap')
        ax1.set_title(f'Accuracy Gap (Train - Val)\nFinal gap: {self.acc_gap[-1]:.4f}')
        ticks = sorted(set([t for t in ax1.get_xticks() if 1 <= t <= max(epochs)] + [max_acc_gap_epoch]))
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([f'*{int(t)}' if t == max_acc_gap_epoch else str(int(t)) for t in ticks])
        self._smart_annotate(
            ax1, max_acc_gap_epoch, self.acc_gap[max_acc_gap_epoch - 1],
            f'max: {self.acc_gap[max_acc_gap_epoch-1]:.3f}',
            '#e74c3c', x_range=max(epochs),
            offset_right=(8, 4), offset_left=(-8, 4)
        )
        ax1.legend()
        ax1.grid(axis='both', alpha=0.3)

        # Loss gap
        ax2.plot(epochs, self.loss_gap, linewidth=2, color='#8e44ad', label='Val - Train Loss')
        ax2.fill_between(epochs, 0, self.loss_gap, alpha=0.15, color='#8e44ad')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        ax2.axhspan(-0.05, 0.1, color='green', alpha=0.06, label='Healthy zone')
        ax2.axvline(x=max_loss_gap_epoch, color='#8e44ad', linestyle=':', alpha=0.7, label='Max gap')
        ax2.set_xlim(1, max(epochs))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Gap')
        ax2.set_title(f'Loss Gap (Val - Train)\nFinal gap: {self.loss_gap[-1]:.4f}')
        ticks = sorted(set([t for t in ax2.get_xticks() if 1 <= t <= max(epochs)] + [max_loss_gap_epoch]))
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f'*{int(t)}' if t == max_loss_gap_epoch else str(int(t)) for t in ticks])
        self._smart_annotate(
            ax2, max_loss_gap_epoch, self.loss_gap[max_loss_gap_epoch - 1],
            f'max: {self.loss_gap[max_loss_gap_epoch-1]:.3f}',
            '#8e44ad', x_range=max(epochs),
            offset_right=(8, 4), offset_left=(-8, 4)
        )
        ax2.legend()
        ax2.grid(axis='both', alpha=0.3)

        plt.suptitle('Overfitting Analysis')
        self._save_fig('train_val_gap.png')

        print(f"\nOverfitting analysis:")
        print(f"  Best epoch (min val_loss):  {best_epoch}")
        print(f"  Acc gap  at best epoch:     {self.acc_gap[best_epoch-1]:.4f}")
        print(f"  Loss gap at best epoch:     {self.loss_gap[best_epoch-1]:.4f}")
        print(f"  Final acc gap:              {self.acc_gap[-1]:.4f}")
        print(f"  Final loss gap:             {self.loss_gap[-1]:.4f}")

    # ── summary ──────────────────────────────────────────────

    def generate_summary(self, mode: str) -> None:
        h = self.history.history if self.history is not None else {}

        d = self.device_info
        if d.get('gpu_count', 0) == 0:
            device_rows = [('Device', 'CPU')]
        else:
            g         = d['gpus'][0]
            mem_total = g.get('memory_total_mb')
            mem_free  = g.get('memory_free_mb')
            mem_str   = f"{mem_total} MB total / {mem_free} MB free" if mem_total else 'n/a'
            device_rows = [
                ('Device',       g.get('smi_name', g.get('device_name', 'GPU'))),
                ('GPU count',    str(d['gpu_count'])),
                ('GPU memory',   mem_str),
                ('Compute cap.', str(g.get('compute_capability', 'n/a'))),
            ]

        if h:
            summary_data = [
                ('Model',    self.config['model']),
                ('Strategy', self.config['strategy']),
                None,
                ('Class weights',   self.config['class_weights']['mode'] if self.config['class_weights']['enabled'] else 'off'),
                ('Label smoothing', f"{self.config['label_smoothing']['value']:.2f}" if self.config['label_smoothing']['enabled'] else 'off'),
                None,
                *device_rows,
                None,
                ('Fit start',   self.fit_start.strftime('%Y-%m-%d %H:%M:%S') if self.fit_start   else 'n/a'),
                ('Fit stop',    self.fit_stop.strftime('%Y-%m-%d %H:%M:%S')  if self.fit_stop    else 'n/a'),
                ('Fit elapsed', self._fmt_duration(self.fit_elapsed)         if self.fit_elapsed is not None else 'n/a'),
                None,
                ('Epochs configured',        self.config['epochs']),
                ('Epochs run',               self.epochs_run if self.epochs_run is not None else 'n/a'),
                ('Early stopping triggered', str(self.epochs_run < self.config['epochs']) if self.epochs_run is not None else 'n/a'),
                None,
                ('Best epoch (min val_loss)', self.best_epoch),
                ('Best val accuracy',         f"{self.best_val_accuracy:.4f}" if self.best_val_accuracy is not None else 'n/a'),
                ('Best val loss',             f"{self.best_val_loss:.4f}"     if self.best_val_loss     is not None else 'n/a'),
                ('Final train accuracy',      f"{h.get('accuracy', h.get('acc', [None]))[-1]:.4f}"),
                ('Final train loss',          f"{h.get('loss', [None])[-1]:.4f}"),
                None,
                ('Acc gap at best epoch',  f"{self.acc_gap[self.best_epoch-1]:.4f}"  if self.acc_gap  is not None and self.best_epoch else 'n/a'),
                ('Loss gap at best epoch', f"{self.loss_gap[self.best_epoch-1]:.4f}" if self.loss_gap is not None and self.best_epoch else 'n/a'),
                ('Final acc gap',          f"{self.acc_gap[-1]:.4f}"                 if self.acc_gap  is not None else 'n/a'),
                ('Final loss gap',         f"{self.loss_gap[-1]:.4f}"                if self.loss_gap is not None else 'n/a'),
            ]
        else:
            summary_data = [
                ('Model',  self.config['model']),
                ('Status', 'not trained yet'),
                None,
                *device_rows,
            ]

        if mode == 'latex':
            self._generate_latex_summary('TrainingHandler', summary_data, 'training_summary.tex')
        else:
            self._generate_ascii_summary('TrainingHandler', summary_data)