from dataclasses import dataclass, field, asdict, fields
from typing import Optional
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


class TrainingMetricsMixin:
    """Mixin for TrainingHandler."""

    def to_metrics_dict(self) -> dict:
        h = self.history.history if self.history is not None else {}

        acc_gap_at_best  = None
        loss_gap_at_best = None
        if self.acc_gap is not None and self.best_epoch is not None:
            acc_gap_at_best  = float(self.acc_gap[self.best_epoch - 1])
        if self.loss_gap is not None and self.best_epoch is not None:
            loss_gap_at_best = float(self.loss_gap[self.best_epoch - 1])

        d          = self.device_info
        gpu_name   = None
        gpu_mem_mb = None
        if d.get('gpus'):
            g          = d['gpus'][0]
            gpu_name   = g.get('smi_name', g.get('device_name'))
            gpu_mem_mb = g.get('memory_total_mb')

        cw = self.config.get('class_weights', {})
        ls = self.config.get('label_smoothing', {})

        return {
            # timing
            'timestamp_start':  self.fit_start.strftime('%Y-%m-%d %H:%M:%S') if self.fit_start else None,
            'timestamp_stop':   self.fit_stop.strftime('%Y-%m-%d %H:%M:%S')  if self.fit_stop  else None,
            'elapsed_seconds':  self.fit_elapsed,

            # training config
            'epochs_configured':  self.config.get('epochs'),
            'class_weights_mode': cw.get('mode') if cw.get('enabled') else 'off',
            'label_smoothing':    ls.get('value') if ls.get('enabled') else None,

            # device
            'device':              d.get('device', 'CPU'),
            'gpu_count':           d.get('gpu_count', 0),
            'gpu_name':            gpu_name,
            'gpu_memory_total_mb': gpu_mem_mb,

            # training results
            'actual_epochs':            self.epochs_run,
            'best_epoch':               self.best_epoch,
            'best_val_accuracy':        self.best_val_accuracy,
            'best_val_loss':            self.best_val_loss,
            'early_stopping_triggered': self.early_stopped,
            'final_train_accuracy':     float(h['accuracy'][-1])  if h.get('accuracy') else None,
            'final_train_loss':         float(h['loss'][-1])      if h.get('loss')     else None,
            'train_val_gap':            self.train_val_gap,
            'acc_gap_at_best_epoch':    acc_gap_at_best,
            'loss_gap_at_best_epoch':   loss_gap_at_best,
        }


# ──────────────────────────────────────────────────────────────────────────────


class ModelMetricsMixin:
    """Mixin for ModelHandler."""
 
    def to_metrics_dict(self) -> dict:
        if self.model is None:
            raise RuntimeError(
                "No model found. Call build() before to_metrics_dict()."
            )
 
        params_total     = int(sum(np.prod(v.shape) for v in self.model.weights))
        params_trainable = int(sum(np.prod(v.shape) for v in self.model.trainable_weights))
        params_frozen    = params_total - params_trainable
 
        layers_total     = len(self.model.layers)
        layers_trainable = sum(1 for l in self.model.layers if l.trainable)
        layers_frozen    = layers_total - layers_trainable
 
        return {
            'model_name':            self.config.get('model'),
            'strategy':              self.config.get('strategy'),
            'input_shape':           list(self.dataset_handler.input_shape),
            'class_num':             self.dataset_handler.class_num,
            'learning_rate':         self.config.get('learning_rate'),
            'dropout_dense':         self.config.get('dropout_dense'),
            'dense_units':           self.config.get('dense_units'),
            'weight_decay':          self.config.get('weight_decay'),
 
            # parameter counts
            'model_params_total':     params_total,
            'model_params_trainable': params_trainable,
            'model_params_frozen':    params_frozen,
 
            # layer counts
            'model_layers_total':     layers_total,
            'model_layers_trainable': layers_trainable,
            'model_layers_frozen':    layers_frozen,
        }


# ──────────────────────────────────────────────────────────────────────────────


class EvaluationMetricsMixin:
    """Mixin for EvaluationHandler."""

    def to_metrics_dict(self) -> dict:
        if self.y_true is None or self.y_pred is None or self.y_pred_proba is None:
            raise RuntimeError(
                "No predictions found. Call evaluate() and predict() before to_metrics_dict()."
            )

        report = self._get_classification_report()
        labels = self.dataset_handler.class_labels
        n_cls  = self.dataset_handler.class_num

        per_class_f1        = {l: report[l]['f1-score']  for l in labels}
        per_class_precision = {l: report[l]['precision'] for l in labels}
        per_class_recall    = {l: report[l]['recall']    for l in labels}
        per_class_accuracy  = {
            labels[i]: float(self.per_class_acc[i]) for i in range(n_cls)
        }

        y_bin     = label_binarize(self.y_true, classes=range(n_cls))
        macro_auc = float(roc_auc_score(y_bin, self.y_pred_proba, average='macro'))
        roc_auc_per_class = {
            labels[i]: float(roc_auc_score(y_bin[:, i], self.y_pred_proba[:, i]))
            for i in range(n_cls)
        }

        ece   = round(self._compute_ece(),         6)
        brier = round(self._compute_brier_score(), 6)
        conf  = self._compute_confidence_stats()

        best_idx  = int(np.argmax(self.per_class_acc))
        worst_idx = int(np.argmin(self.per_class_acc))
        best_class  = f"{labels[best_idx]}  ({self.per_class_acc[best_idx]:.4f})"
        worst_class = f"{labels[worst_idx]} ({self.per_class_acc[worst_idx]:.4f})"

        misclassified      = int((self.y_pred != self.y_true).sum())
        misclassified_rate = round(float((self.y_pred != self.y_true).mean()), 6)

        return {
            # scalars
            'test_accuracy':   self.test_accuracy,
            'test_loss':       self.test_loss,
            'f1_macro':        report['macro avg']['f1-score'],
            'f1_weighted':     report['weighted avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro':    report['macro avg']['recall'],
            'macro_auc':       macro_auc,
            'ece':             ece,
            'brier_score':     brier,
            'n_test_samples':  len(self.y_true),

            # per-class
            'class_names':         list(labels),
            'confusion_matrix':    self.confusion_matrix.tolist(),
            'per_class_f1':        per_class_f1,
            'per_class_precision': per_class_precision,
            'per_class_recall':    per_class_recall,
            'per_class_accuracy':  per_class_accuracy,
            'roc_auc_per_class':   roc_auc_per_class,

            # summary
            'best_class':               best_class,
            'worst_class':              worst_class,
            'total_misclassifications': misclassified,
            'misclassification_rate':   misclassified_rate,

            # confidence
            'confidence_mean_overall':   conf['mean_overall'],
            'confidence_mean_correct':   conf['mean_correct'],
            'confidence_mean_incorrect': conf['mean_incorrect'],
            'confidence_mean_per_class': conf['mean_per_class'],
        }


# ──────────────────────────────────────────────────────────────────────────────

class TFLiteMetricsMixin:
    """
    Mixin for TFLiteHandler.

    Requires attributes set by register_from_evaluation() + convert_all() + benchmark_all():
        self.model, self.keras_model, self.benchmark_results,
        self.conversion_times, self.model_sizes
    """

    def to_metrics_dict(self) -> dict:
        if self.keras_model is None:
            raise RuntimeError(
                "No Keras model data."
            )
        if not self.benchmark_results:
            raise RuntimeError(
                "No benchmark results."
            )

        params_total     = self.model.count_params()
        params_trainable = int(sum(
            np.prod(w.shape) for w in self.model.trainable_weights
        ))
        size_keras_kb = self.keras_model.get(
            'file_size_kb', self.keras_model.get('model_size_kb')
        )

        tflite = {}
        for variant_key, r in self.benchmark_results.items():
            tflite[variant_key] = {
                'model_type':              r['model_type'],
                'accuracy':                r['accuracy'],
                'accuracy_delta_vs_keras': r['accuracy_delta_vs_keras'],
                'f1_macro':                r['f1_macro'],
                'f1_weighted':             r['f1_weighted'],
                'per_class_accuracy':      r['per_class_accuracy'],
                'per_class_f1':            r['per_class_f1'],
                'per_class_precision':     r['per_class_precision'],
                'per_class_recall':        r['per_class_recall'],
                'confusion_matrix':        r['confusion_matrix'],
                'mean_inference_time_ms':  r['mean_inference_time_ms'],
                'std_inference_time_ms':   r['std_inference_time_ms'],
                'p95_inference_time_ms':   r['p95_inference_time_ms'],
                'model_size_kb':           r['model_size_kb'],
                'compression_ratio':       r['compression_ratio'],
                'samples_tested':          r['samples_tested'],
                'conversion_time_s':       self.conversion_times.get(variant_key),
                'confidence':              r.get('confidence'),  # None for int8_quant
            }

        return {
            'model_params_total':     params_total,
            'model_params_trainable': params_trainable,
            'model_size_keras_kb':    size_keras_kb,
            'tflite':                 tflite,
        }

 
@dataclass
class ExperimentMetrics:
 
    # ── identification ────────────────────────────────────────────────────────
    experiment_id:   str
    model_name:      Optional[str]
    dataset:         Optional[str]
    strategy:        Optional[str]
    timestamp_start: Optional[str]   = None
    timestamp_stop:  Optional[str]   = None
    elapsed_seconds: Optional[float] = None

    # ── model info ────────────────────────────────────────────────────────────
    model_params_total:     Optional[int]   = None
    model_params_trainable: Optional[int]   = None
    model_params_frozen:    Optional[int]   = None
    model_size_keras_kb:    Optional[float] = None
    model_layers_total:     Optional[int]   = None
    model_layers_trainable: Optional[int]   = None
    model_layers_frozen:    Optional[int]   = None
    input_shape:            Optional[list]  = None
    learning_rate:          Optional[float] = None
    dropout_dense:          Optional[float] = None
    dense_units:            Optional[int]   = None
    weight_decay:           Optional[float] = None
 
 
    # ── training config ───────────────────────────────────────────────────────
    epochs_configured:  Optional[int]   = None
    class_weights_mode: Optional[str]   = None
    label_smoothing:    Optional[float] = None
 
    # ── device ────────────────────────────────────────────────────────────────
    device:              Optional[str] = None
    gpu_count:           Optional[int] = None
    gpu_name:            Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
 
    # ── training ──────────────────────────────────────────────────────────────
    actual_epochs:            Optional[int]   = None
    best_epoch:               Optional[int]   = None
    best_val_accuracy:        Optional[float] = None
    best_val_loss:            Optional[float] = None
    early_stopping_triggered: Optional[bool]  = None
    final_train_accuracy:     Optional[float] = None
    final_train_loss:         Optional[float] = None
    train_val_gap:            Optional[float] = None
    acc_gap_at_best_epoch:    Optional[float] = None
    loss_gap_at_best_epoch:   Optional[float] = None
 
    # ── evaluation ────────────────────────────────────────────────────────────
    test_accuracy:   Optional[float] = None
    test_loss:       Optional[float] = None
    f1_macro:        Optional[float] = None
    f1_weighted:     Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro:    Optional[float] = None
    macro_auc:       Optional[float] = None
 
    per_class_f1:        dict = field(default_factory=dict)
    per_class_precision: dict = field(default_factory=dict)
    per_class_recall:    dict = field(default_factory=dict)
    per_class_accuracy:  dict = field(default_factory=dict)
    roc_auc_per_class:   dict = field(default_factory=dict)
    confusion_matrix:    list = field(default_factory=list)
    class_names:         list = field(default_factory=list)
    n_test_samples:      Optional[int] = None
 
    best_class:  Optional[str] = None
    worst_class: Optional[str] = None
    total_misclassifications: Optional[int]   = None
    misclassification_rate:   Optional[float] = None
 
    ece:         Optional[float] = None
    brier_score: Optional[float] = None
 
    # ── confidence ────────────────────────────────────────────────────────────
    confidence_mean_overall:   Optional[float] = None
    confidence_mean_correct:   Optional[float] = None
    confidence_mean_incorrect: Optional[float] = None
    confidence_mean_per_class: dict = field(default_factory=dict)

    # ── tflite — keys: 'float32', 'dynamic_quant', 'int8_quant' ──────────────
    tflite: dict = field(default_factory=dict)
 
    # ── update API ────────────────────────────────────────────────────────────
 
    def update(self, *,
               model:      Optional[dict] = None,
               training:   Optional[dict] = None,
               evaluation: Optional[dict] = None,
               tflite:     Optional[dict] = None) -> 'ExperimentMetrics':
        """Accept dicts from handler.to_metrics_dict(). Each argument is optional.
        Returns self to allow chaining."""
        if model      is not None: self._apply(model)
        if training   is not None: self._apply(training)
        if evaluation is not None: self._apply(evaluation)
        if tflite     is not None: self._apply(tflite)
        return self
 
    def _apply(self, data: dict) -> None:
        """Write dict values into matching dataclass fields; unknown keys are ignored."""
        known = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in known:
                setattr(self, key, value)
 
    # ── serialization ─────────────────────────────────────────────────────────
 
    def to_dict(self) -> dict:
        """Nested dict — used as the archive JSON format."""
        return asdict(self)
 
    def to_flat_dict(self) -> dict:
        """Flat dict suitable as a single pd.DataFrame row in nb02/nb03."""
        _SKIP_TOP = {
            'per_class_f1', 'per_class_precision', 'per_class_recall',
            'per_class_accuracy', 'roc_auc_per_class', 'confusion_matrix',
            'confidence_mean_per_class', 'tflite',
        }
        _SKIP_TFLITE = {
            'per_class_accuracy', 'per_class_f1', 'per_class_precision',
            'per_class_recall', 'confusion_matrix', 'confidence',
        }
 
        flat = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _SKIP_TOP
        }
 
        for variant, metrics in self.tflite.items():
            if not isinstance(metrics, dict):
                continue
            prefix = f"tflite_{variant}"
            for key, val in metrics.items():
                if key not in _SKIP_TFLITE:
                    flat[f"{prefix}_{key}"] = val
 
        return flat
 
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
 
    @classmethod
    def load(cls, path: str) -> 'ExperimentMetrics':
        with open(path, 'r') as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})