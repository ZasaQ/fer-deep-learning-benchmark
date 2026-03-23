import os
import time
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix as sk_cm,
)

from .BaseHandler import BaseHandler
from .DatasetHandler import DatasetHandler
from .DataAugmentationHandler import DataAugmentationHandler
from .EvaluationHandler import EvaluationHandler


class TFLiteHandler(BaseHandler):
    """Handles TensorFlow Lite conversion, benchmarking and mobile deployment visualization."""

    _COLOR_MAP = {
        'float32':       'steelblue',
        'dynamic_quant': 'orange',
        'int8_quant':    'green',
    }
    _LABEL_MAP = {
        'float32':       'Float32',
        'dynamic_quant': 'Dynamic Quant',
        'int8_quant':    'INT8 Quant',
    }

    def __init__(self,
                 config: dict,
                 model: tf.keras.Model,
                 dataset_handler: DatasetHandler,
                 data_augmentation_handler: DataAugmentationHandler,
                 evaluation_handler: EvaluationHandler,
                 keras_model_path: str,
                 root_directory: str,
                 visualizations_directory: str):
        self.config                    = config
        self.model                     = model
        self.dataset_handler           = dataset_handler
        self.data_augmentation_handler = data_augmentation_handler
        self.keras_model_path          = keras_model_path
        self.root_directory            = root_directory
        self.evaluation_handler        = evaluation_handler
        super().__init__(visualizations_directory)

        self.class_names: List[str]  = dataset_handler.class_names
        self.class_labels: List[str] = dataset_handler.class_labels

        self.tflite_model:         Optional[bytes] = None
        self.tflite_quant_dynamic: Optional[bytes] = None
        self.tflite_quant_int8:    Optional[bytes] = None

        self.keras_model: Optional[dict] = None

        self.conversion_times:  dict = {}
        self.model_sizes:       dict = {}
        self.benchmark_results: dict = {}

        print('TFLiteHandler has been initialized.')

    # ── private helpers ──────────────────────────────────────

    @property
    def _model_map(self) -> dict:
        return {
            'float32':       self.tflite_model,
            'dynamic_quant': self.tflite_quant_dynamic,
            'int8_quant':    self.tflite_quant_int8,
        }

    def _style_lists(self, keys: List[str]) -> Tuple[List[str], List[str]]:
        labels = [self._LABEL_MAP[k] for k in keys]
        colors = [self._COLOR_MAP[k] for k in keys]
        return labels, colors

    def _prepare_test_generator(self, shuffle: bool) -> object:
        test_generator = self.data_augmentation_handler.test_generator
        if hasattr(test_generator, 'index_array') and test_generator.index_array is not None:
            if shuffle:
                np.random.shuffle(test_generator.index_array)
            else:
                labels_for_indices = test_generator.classes[test_generator.index_array]
                interleaved = []
                class_buckets = [
                    list(test_generator.index_array[labels_for_indices == c])
                    for c in np.unique(labels_for_indices)
                ]
                while any(class_buckets):
                    for bucket in class_buckets:
                        if bucket:
                            interleaved.append(bucket.pop(0))
                test_generator.index_array = np.array(interleaved)
            test_generator.index = 0
        return test_generator

    def _total_test_samples(self) -> int:
        gen = self.data_augmentation_handler.test_generator
        if hasattr(gen, 'samples'):
            return gen.samples
        if hasattr(gen, 'n'):
            return gen.n
        return len(gen) * getattr(gen, 'batch_size', 32)

    def _compute_confidence_stats(self,
                                   y_pred_proba: np.ndarray,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray) -> dict:
        confidence = np.max(y_pred_proba, axis=1)
        correct    = (y_pred == y_true)
        return {
            'mean_overall':   float(np.mean(confidence)),
            'mean_correct':   float(np.mean(confidence[correct]))   if correct.any()   else None,
            'mean_incorrect': float(np.mean(confidence[~correct]))  if (~correct).any() else None,
            'mean_per_class': {
                self.class_labels[i]: float(np.mean(confidence[y_true == i]))
                if np.any(y_true == i) else None
                for i in range(len(self.class_labels))
            },
        }

    def register_keras_model(self, accuracy: Optional[float] = None) -> None:
        """Register Keras model accuracy and size."""
        if accuracy is None:
            if self.evaluation_handler is None or self.evaluation_handler.test_accuracy is None:
                raise RuntimeError(
                    "No accuracy provided and no EvaluationHandler available. "
                    "Pass accuracy= or set evaluation_handler in __init__()."
                )
            accuracy = self.evaluation_handler.test_accuracy

        params  = self.model.count_params()
        size_kb = (params * 4) / 1024

        self.keras_model = {
            'accuracy':               float(accuracy),
            'params':                 params,
            'model_size_kb':          round(size_kb, 2),
            'mean_inference_time_ms': None,
            'std_inference_time_ms':  None,
            'p95_inference_time_ms':  None,
            'samples_tested':         None,
            'per_class_accuracy':     None,
            'test_loss':              None,
        }

        if self.keras_model_path and os.path.exists(self.keras_model_path):
            file_size_kb = os.path.getsize(self.keras_model_path) / 1024
            self.keras_model['file_size_kb'] = round(file_size_kb, 2)
            self.keras_model['model_size_kb'] = round(file_size_kb, 2)

        print('Keras model registered:')
        print(f'  Accuracy:      {accuracy:.4f}')
        print(f'  Params:        {params:,}')
        print(f'  Size (params): {size_kb:.2f} KB')
        if 'file_size_kb' in self.keras_model:
            print(f'  Size (file):   {self.keras_model["file_size_kb"]:.2f} KB  <- used for comparisons')

    def register_from_evaluation(self) -> None:
        """Register Keras model metrics."""
        ev = self.evaluation_handler
        if ev is None:
            raise RuntimeError("No EvaluationHandler available.")
        if ev.test_accuracy is None:
            raise RuntimeError("Call evaluation_handler.evaluate() first.")
        if ev.y_true is None or ev.per_class_acc is None:
            raise RuntimeError("Call evaluation_handler.predict() first.")

        self.register_keras_model(accuracy=ev.test_accuracy)

        per_class_accuracy = {
            ev.dataset_handler.class_labels[i]: float(ev.per_class_acc[i])
            for i in range(len(ev.dataset_handler.class_labels))
        }
        self.keras_model['per_class_accuracy'] = per_class_accuracy
        self.keras_model['test_loss']          = ev.test_loss

        print(f'  Per-class accuracy registered for {len(per_class_accuracy)} classes.')

    # ── conversion ───────────────────────────────────────────

    def convert_float32(self) -> bytes:
        print('Converting to TFLite - float32...')
        start = time.time()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.tflite_model = converter.convert()
        self.conversion_times['float32'] = time.time() - start
        self.model_sizes['float32'] = len(self.tflite_model)
        return self.tflite_model

    def convert_dynamic_range_quant(self) -> bytes:
        print('Converting to TFLite - dynamic range quantization...')
        start = time.time()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.tflite_quant_dynamic = converter.convert()
        self.conversion_times['dynamic_quant'] = time.time() - start
        self.model_sizes['dynamic_quant'] = len(self.tflite_quant_dynamic)
        return self.tflite_quant_dynamic

    def convert_int8_quant(self, int8_calibration_fraction: float) -> bytes:
        """Convert to INT8 quantization using a representative dataset."""
        print('Converting to TFLite - INT8 quantization...')
        start = time.time()

        train_gen   = self.data_augmentation_handler.train_generator
        train_total = getattr(train_gen, 'samples', getattr(train_gen, 'n',
                              len(train_gen) * train_gen.batch_size))
        num_samples = max(100, int(train_total * int8_calibration_fraction))
        batch_size  = getattr(train_gen, 'batch_size', 32)
        max_batches = max(1, num_samples // batch_size)

        def representative_data_gen():
            for i, (images, _) in enumerate(train_gen):
                if i >= max_batches:
                    break
                for img in images:
                    yield [np.expand_dims(img, axis=0).astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        self.tflite_quant_int8 = converter.convert()
        self.conversion_times['int8_quant'] = time.time() - start
        self.model_sizes['int8_quant'] = len(self.tflite_quant_int8)
        print(f'INT8 calibration: {num_samples} samples '
              f'({int8_calibration_fraction * 100:.0f}% of {train_total} train samples)')
        return self.tflite_quant_int8

    def convert_all(self, int8_calibration_fraction: float) -> None:
        self.convert_float32()
        self.convert_dynamic_range_quant()
        self.convert_int8_quant(int8_calibration_fraction=int8_calibration_fraction)

    # ── saving ───────────────────────────────────────────────

    def save_tflite(self, filepath: str, model_type: str) -> None:
        tflite_model = self._model_map.get(model_type)
        if tflite_model is None:
            raise ValueError(f"Model type '{model_type}' not converted yet.")
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
        print(f'TFLite model saved to: {filepath}')

    def save_all(self) -> dict:
        model_name = (
            f"{self.config['dataset']}_"
            f"{self.config['model']}_"
            f"{self.config['strategy']}"
        )

        filepaths = {}
        for model_type, suffix in [
            ('float32',       'float32'),
            ('dynamic_quant', 'dynamic_quant'),
            ('int8_quant',    'int8_quant'),
        ]:
            if self._model_map[model_type] is not None:
                path = os.path.join(self.root_directory, f'{model_name}_{suffix}.tflite')
                self.save_tflite(path, model_type)
                filepaths[model_type] = path
        return filepaths

    # ── benchmarking ─────────────────────────────────────────

    def benchmark_keras_inference(self, shuffle: bool) -> dict:
        if self.keras_model is None:
            raise RuntimeError("Call register_from_evaluation() first.")

        total       = self._total_test_samples()
        test_gen    = self._prepare_test_generator(shuffle=shuffle)
        max_batches = len(test_gen)
        times       = []
        processed   = 0

        print(f'Benchmarking Keras model inference on full test set ({total} samples)...')

        for batch_idx, (images, _) in enumerate(test_gen):
            if batch_idx >= max_batches:
                break
            for img in images:
                inp   = np.expand_dims(img, axis=0)
                start = time.time()
                self.model.predict(inp, verbose=0)
                times.append((time.time() - start) * 1000)
                processed += 1
            if processed % 100 == 0:
                print(f'  {processed}/{total} samples...', end='\r')

        if not times:
            raise RuntimeError("No samples were processed.")

        self.keras_model.update({
            'mean_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms':  float(np.std(times)),
            'p95_inference_time_ms':  float(np.percentile(times, 95)),
            'samples_tested':         processed,
        })

        print(f'\n  Samples tested:  {processed}')
        print(f'  Mean inference:  {self.keras_model["mean_inference_time_ms"]:.2f} ms')
        print(f'  Std inference:   {self.keras_model["std_inference_time_ms"]:.2f} ms')
        print(f'  P95 inference:   {self.keras_model["p95_inference_time_ms"]:.2f} ms')
        return self.keras_model

    def benchmark_inference(self, model_type: str, shuffle: bool, save_raw: bool) -> dict:
        tflite_model = self._model_map.get(model_type)
        if tflite_model is None:
            raise ValueError(f"Model type '{model_type}' not converted yet.")

        try:
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
        except RuntimeError as e:
            if "XNNPack" in str(e) or "XNNPACK" in str(e):
                print(f"\nXNNPACK failed for {model_type}. Falling back to default CPU kernels...")
                interpreter = tf.lite.Interpreter(
                    model_content=tflite_model,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
                )
                interpreter.allocate_tensors()
            else:
                raise

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype    = input_details[0]['dtype']
        output_dtype   = output_details[0]['dtype']
        i_scale, i_zero = input_details[0].get('quantization', (1.0, 0))
        o_scale, o_zero = output_details[0].get('quantization', (1.0, 0))

        is_int_output = output_dtype in (np.uint8, np.int8)

        total       = self._total_test_samples()
        test_gen    = self._prepare_test_generator(shuffle=shuffle)
        max_batches = len(test_gen)

        y_true, y_pred, times          = [], [], []
        y_pred_proba_list              = []
        processed = 0

        print(f'Benchmarking {model_type} on full test set ({total} samples)...')

        for batch_idx, (images, labels) in enumerate(test_gen):
            if batch_idx >= max_batches:
                break
            for img, label in zip(images, labels):
                input_data = np.expand_dims(img, axis=0).astype(np.float32)

                if input_dtype in (np.uint8, np.int8):
                    safe_scale = i_scale if i_scale > 0 else 1.0
                    quantized  = input_data / safe_scale + i_zero
                    if input_dtype == np.uint8:
                        input_data = np.clip(quantized, 0, 255).astype(np.uint8)
                    else:
                        input_data = np.clip(quantized, -128, 127).astype(np.int8)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                start = time.time()
                interpreter.invoke()
                times.append((time.time() - start) * 1000)

                output_data = interpreter.get_tensor(output_details[0]['index'])
                if is_int_output:
                    output_data = (output_data.astype(np.float32) - o_zero) * o_scale
                else:
                    y_pred_proba_list.append(output_data[0].copy())

                y_true.append(np.argmax(label))
                y_pred.append(np.argmax(output_data))
                processed += 1

            if processed % 100 == 0:
                print(f'  {processed}/{total} samples...', end='\r')

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        unique_classes, counts = np.unique(y_true, return_counts=True)
        dist_map = dict(zip(unique_classes, counts))
        print(f'\nClass distribution for {model_type}:')
        for i, name in enumerate(self.class_labels):
            print(f'  - {name}: {dist_map.get(i, 0)} samples')

        if len(unique_classes) < len(self.class_labels):
            print(f'WARNING: Only {len(unique_classes)}/{len(self.class_labels)} classes represented.')

        per_class_accuracy = {}
        per_class_f1       = {}
        per_class_precision = {}
        per_class_recall   = {}

        for i in range(len(self.class_labels)):
            label_name = self.class_labels[i]
            mask       = (y_true == i)
            if np.any(mask):
                per_class_accuracy[label_name]  = float(np.mean(y_pred[mask] == y_true[mask]))
                per_class_f1[label_name]        = float(f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0))
                per_class_precision[label_name] = float(precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0))
                per_class_recall[label_name]    = float(recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0))
            else:
                per_class_accuracy[label_name]  = float('nan')
                per_class_f1[label_name]        = float('nan')
                per_class_precision[label_name] = float('nan')
                per_class_recall[label_name]    = float('nan')

        keras_size_kb = None
        keras_acc     = None
        if self.keras_model is not None:
            keras_size_kb = self.keras_model.get('file_size_kb', self.keras_model.get('model_size_kb'))
            keras_acc     = self.keras_model.get('accuracy')

        size_kb          = self.model_sizes[model_type] / 1024
        compression_ratio = round(keras_size_kb / size_kb, 3) if keras_size_kb else None
        accuracy_val      = accuracy_score(y_true, y_pred)
        accuracy_delta    = round(accuracy_val - keras_acc, 6) if keras_acc is not None else None

        results = {
            'model_type':             model_type,
            'accuracy':               accuracy_val,
            'accuracy_delta_vs_keras': accuracy_delta,
            'f1_macro':               float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
            'f1_weighted':            float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'per_class_accuracy':     per_class_accuracy,
            'per_class_f1':           per_class_f1,
            'per_class_precision':    per_class_precision,
            'per_class_recall':       per_class_recall,
            'confusion_matrix':       sk_cm(y_true, y_pred).tolist(),
            'mean_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms':  float(np.std(times)),
            'p95_inference_time_ms':  float(np.percentile(times, 95)),
            'model_size_kb':          round(size_kb, 5),
            'compression_ratio':      compression_ratio,
            'samples_tested':         processed,
        }

        if not is_int_output and y_pred_proba_list:
            y_pred_proba = np.array(y_pred_proba_list)
            results['confidence'] = self._compute_confidence_stats(y_pred_proba, y_true, y_pred)
        else:
            results['confidence'] = None

        if save_raw:
            results['raw_inference_times_ms'] = times

        print(f'  Accuracy:        {results["accuracy"]:.4f}')
        print(f'  F1 macro:        {results["f1_macro"]:.4f}')
        print(f'  Mean inference:  {results["mean_inference_time_ms"]:.2f} ms')
        print(f'  P95 inference:   {results["p95_inference_time_ms"]:.2f} ms')
        print(f'  Compression:     {compression_ratio}x' if compression_ratio else '  Compression:     n/a')
        print(f'  Samples tested:  {processed}')

        self.benchmark_results[model_type] = results
        return results

    def benchmark_all(self, shuffle: bool, save_raw: bool) -> None:
        total = self._total_test_samples()
        print(f'\n{"=" * 60}\nBENCHMARKING ALL VARIANTS — full test set ({total} samples)\n{"=" * 60}')
        for model_type, model_bytes in self._model_map.items():
            if model_bytes:
                self.data_augmentation_handler.reset_test_generator()
                self.benchmark_inference(model_type, shuffle=shuffle, save_raw=save_raw)

    # ── visualizations ───────────────────────────────────────

    def plot_inference_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        has_raw = any('raw_inference_times_ms' in r for r in self.benchmark_results.values())
        if not self._guard(has_raw,
                           'No raw times found. Call benchmark_all(save_raw=True) first.'):
            return

        keys_with_raw  = [k for k, r in self.benchmark_results.items()
                          if 'raw_inference_times_ms' in r]
        labels, colors = self._style_lists(keys_with_raw)
        data           = [self.benchmark_results[k]['raw_inference_times_ms'] for k in keys_with_raw]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bp = ax1.boxplot(data, patch_artist=True, notch=False,
                         showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median_line in bp['medians']:
            median_line.set_color('black')
            median_line.set_linewidth(2)

        for i, d in enumerate(data, start=1):
            p50    = np.percentile(d, 50)
            p95    = np.percentile(d, 95)
            offset = max((p95 - p50) * 0.1, 0.05)
            ax1.text(i + 0.3, p50 - offset, f'P50: {p50:.1f}ms', va='top',    fontsize=7)
            ax1.text(i + 0.3, p95 + offset, f'P95: {p95:.1f}ms', va='bottom', fontsize=7, color='dimgray')

        ax1.set_xticks(range(1, len(labels) + 1))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title('Inference Time Box Plot')
        ax1.grid(axis='y', alpha=0.3)

        parts = ax2.violinplot(data, positions=range(1, len(data) + 1),
                               showmeans=False, showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        for key in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
            if key in parts:
                parts[key].set_color('black')

        for i, (d, color) in enumerate(zip(data, colors), start=1):
            jitter = np.random.uniform(-0.08, 0.08, size=len(d))
            ax2.scatter(np.full(len(d), i) + jitter, d,
                        color=color, alpha=0.35, s=8, zorder=2)

        ax2.set_xticks(range(1, len(labels) + 1))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Inference Time Violin Plot')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Inference Time Distribution')
        self._save_fig('inference_distribution.png')

    def plot_radar_chart(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        if not self._guard(bool(self.benchmark_results),
                           'No benchmark results. Call benchmark_all() first.'):
            return

        has_keras     = self.keras_model is not None
        has_keras_inf = has_keras and self.keras_model.get('mean_inference_time_ms') is not None

        keras_size_kb = (
            self.keras_model.get('file_size_kb', self.keras_model['model_size_kb'])
            if has_keras else None
        )
        baseline_size = (keras_size_kb * 1024) if keras_size_kb else self.model_sizes.get('float32', 1)

        def _raw_vals(r: dict) -> list:
            return [
                r['accuracy'],
                baseline_size / (r['model_size_kb'] * 1024),
                1.0 / max(r['mean_inference_time_ms'], 1e-6) if r.get('mean_inference_time_ms') else None,
                1.0 / r['model_size_kb'],
            ]

        entries = {}
        if has_keras:
            entries['keras'] = _raw_vals({
                'accuracy':               self.keras_model['accuracy'],
                'model_size_kb':          keras_size_kb,
                'mean_inference_time_ms': self.keras_model.get('mean_inference_time_ms'),
            })
        for mt in self.benchmark_results:
            entries[mt] = _raw_vals(self.benchmark_results[mt])

        include_speed = any(v[2] is not None for v in entries.values())
        axis_defs = [
            ('accuracy',    'Accuracy',           0, +1, (0.0, 1.0)),
            ('compression', 'Compression\nRatio', 1, +1, (None, None)),
            ('size',        'Size\nEfficiency',   3, +1, (None, None)),
        ]
        if include_speed:
            axis_defs.insert(2, ('speed', 'Speed\n(ms⁻¹)', 2, +1, (None, None)))

        N = len(axis_defs)

        normalized = {k: [] for k in entries}
        for _, _, raw_idx, direction, fixed_range in axis_defs:
            vals = [v[raw_idx] for v in entries.values() if v[raw_idx] is not None]
            lo, hi = fixed_range
            if lo is None:
                data_min = min(vals); data_max = max(vals)
                span = max(data_max - data_min, 1e-9)
                lo = data_min - span * 0.1; hi = data_max + span * 0.1
            elif hi is None:
                data_max = max(vals)
                span = max(data_max - lo, 1e-9)
                hi = lo + span * 1.2
            for k, v in entries.items():
                val = v[raw_idx]
                if val is None:
                    normalized[k].append(0.0)
                else:
                    norm = (val - lo) / (hi - lo)
                    if direction == -1:
                        norm = 1.0 - norm
                    normalized[k].append(float(np.clip(norm, 0.0, 1.0)))

        color_map = {'keras': '#444444', **self._COLOR_MAP}
        label_map = {'keras': 'Keras',   **self._LABEL_MAP}

        angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig    = plt.figure(figsize=figsize)
        ax     = fig.add_subplot(121, polar=True)
        ax_box = fig.add_subplot(122)
        ax_box.axis('off')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 1)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='grey')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label for _, label, *_ in axis_defs], fontsize=11)

        for k, norm_vals in normalized.items():
            color = color_map[k]
            vals  = norm_vals + norm_vals[:1]
            lw    = 3.0 if k == 'keras' else 2.5
            ls    = '--' if k == 'keras' else '-'
            ax.plot(angles, vals, color=color, linewidth=lw,
                    linestyle=ls, label=label_map[k])
            ax.fill(angles, vals, color=color, alpha=0.08 if k == 'keras' else 0.15)
            ax.scatter(angles[:-1], norm_vals, color=color, s=80,
                       zorder=5, edgecolors='white',
                       marker='s' if k == 'keras' else 'o')

        ax.set_title('Metric Radar', pad=25)

        metric_labels = [d[0].capitalize() for d in axis_defs]
        raw_idx_list  = [d[2] for d in axis_defs]

        n_rows  = len(entries)
        row_h   = 0.10
        total_h = 0.06 + n_rows * row_h
        y       = 0.5 + total_h / 2
        n_cols  = len(metric_labels)
        col_xs  = np.linspace(0.01, 0.99, n_cols + 1).tolist()

        ax_box.text(col_xs[0], y, 'Model', fontsize=9, fontweight='bold',
                    transform=ax_box.transAxes, va='top', color='black')
        for xi, ml in zip(col_xs[1:], metric_labels):
            ax_box.text(xi, y, ml, fontsize=8, fontweight='bold',
                        transform=ax_box.transAxes, va='top', ha='center', color='black')

        ax_box.plot([0.02, 0.98], [y - 0.06, y - 0.06],
                    color='lightgray', linewidth=0.8,
                    transform=ax_box.transAxes)

        for i, (k, raw_vals) in enumerate(entries.items()):
            color = color_map[k]
            row_y = y - 0.12 - i * 0.10
            ax_box.text(col_xs[0], row_y, label_map[k], fontsize=9, fontweight='bold',
                        transform=ax_box.transAxes, va='top', color=color)
            for xi, ri in zip(col_xs[1:], raw_idx_list):
                val = raw_vals[ri]
                txt = f'{val:.4f}' if val is not None else 'n/a'
                ax_box.text(xi, row_y, txt, fontsize=9,
                            transform=ax_box.transAxes, va='top', ha='center', color=color)

        ax_box.set_title('Metric Values', fontsize=10, fontweight='bold', pad=10)

        plt.suptitle('Keras vs Quantized Variants - Radar Chart',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_fig('radar_chart.png')

    def plot_quantization_error_heatmap(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        """
        Per-class accuracy heatmap for all models.
        """
        has_per_class = any('per_class_accuracy' in r for r in self.benchmark_results.values())
        if not self._guard(has_per_class,
                           'No per-class accuracy found. Call benchmark_all() first.'):
            return

        all_class_names = self.class_labels

        tflite_types = [mt for mt in self.benchmark_results
                        if 'per_class_accuracy' in self.benchmark_results[mt]]

        has_keras_per_class = (
            self.keras_model is not None
            and self.keras_model.get('per_class_accuracy') is not None
        )

        rows = []
        if has_keras_per_class:
            rows.append(('Keras', '#444444', self.keras_model['per_class_accuracy']))
        for mt in tflite_types:
            rows.append((
                self._LABEL_MAP.get(mt, mt),
                self._COLOR_MAP[mt],
                self.benchmark_results[mt]['per_class_accuracy'],
            ))

        row_labels = [r[0] for r in rows]
        row_colors = [r[1] for r in rows]
        acc_matrix = np.array([
            [r[2].get(cn, np.nan) for cn in all_class_names]
            for r in rows
        ])

        if has_keras_per_class:
            baseline_row   = np.array([self.keras_model['per_class_accuracy'].get(cn, np.nan) for cn in all_class_names])
            baseline_label = 'Keras'
            tflite_rows    = [r for r in rows if r[0] != 'Keras']
            tflite_labels  = [r[0] for r in tflite_rows]
            tflite_colors  = [r[1] for r in tflite_rows]
            drop_matrix    = np.array([
                [r[2].get(cn, np.nan) for cn in all_class_names]
                for r in tflite_rows
            ]) - baseline_row
        elif 'float32' in tflite_types:
            float32_idx    = tflite_types.index('float32')
            baseline_row   = acc_matrix[float32_idx]
            baseline_label = 'Float32'
            tflite_labels  = row_labels
            tflite_colors  = row_colors
            drop_matrix    = acc_matrix - baseline_row
        else:
            baseline_row   = None
            baseline_label = None
            tflite_labels  = row_labels
            tflite_colors  = row_colors
            drop_matrix    = None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        im = ax.imshow(acc_matrix, aspect='auto', cmap='RdYlGn', vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Accuracy')
        ax.set_xticks(range(len(all_class_names)))
        ax.set_xticklabels(all_class_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title('Per-Class Accuracy')

        for tick, color in zip(ax.get_yticklabels(), row_colors):
            tick.set_color(color)
            if color == '#444444':
                tick.set_fontweight('bold')

        if has_keras_per_class and len(tflite_types) > 0:
            ax.axhline(y=0.5, color='white', linewidth=3)

        for i in range(len(row_labels)):
            for j in range(len(all_class_names)):
                val = acc_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                            color='black' if 0.25 < val < 0.75 else 'white')

        ax2 = axes[1]
        if drop_matrix is not None:
            absmax = max(np.nanmax(np.abs(drop_matrix)), 1e-6)
            im2    = ax2.imshow(drop_matrix, aspect='auto', cmap='RdBu',
                                vmin=-absmax, vmax=absmax)
            plt.colorbar(im2, ax=ax2, label='Accuracy Delta Difference')
            ax2.set_xticks(range(len(all_class_names)))
            ax2.set_xticklabels(all_class_names, rotation=45, ha='right', fontsize=9)
            ax2.set_yticks(range(len(tflite_labels)))
            ax2.set_yticklabels(tflite_labels, fontsize=10)
            ax2.set_title(f'Per-Class Delta vs {baseline_label}')

            for tick, color in zip(ax2.get_yticklabels(), tflite_colors):
                tick.set_color(color)

            for i in range(len(tflite_labels)):
                for j in range(len(all_class_names)):
                    val = drop_matrix[i, j]
                    if not np.isnan(val):
                        ax2.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=8,
                                 color='white' if abs(val) > absmax * 0.6 else 'black')
        else:
            ax2.axis('off')

        plt.suptitle('Quantization Impact per Emotion Class')
        plt.tight_layout()
        self._save_fig('quantization_error_heatmap.png')

    def plot_full_comparison(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        if not self._guard(self.keras_model is not None,
                           'No Keras model. Call register_from_evaluation() first.'):
            return
        if not self._guard(bool(self.benchmark_results),
                           'No TFLite benchmark results. Call benchmark_all() first.'):
            return

        color_map = {'keras': '#444444', **self._COLOR_MAP}
        label_map = {
            'keras':         'Keras',
            'float32':       'TFLite\nFloat32',
            'dynamic_quant': 'TFLite\nDynamic',
            'int8_quant':    'TFLite\nINT8',
        }

        entries    = [('keras', self.keras_model)] + list(self.benchmark_results.items())
        labels     = [label_map[k] for k, _ in entries]
        colors     = [color_map[k] for k, _ in entries]
        x          = range(len(entries))
        keras_size = self.keras_model.get('file_size_kb', self.keras_model['model_size_kb'])
        keras_acc  = self.keras_model.get('accuracy', 0)

        fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'hspace': 0.4})
        (ax_size, ax_ratio), (ax_acc, ax_time) = axes

        def _keras_hline(ax, val):
            ax.axhline(y=val, color='#444444', linestyle='--', linewidth=1.2,
                       alpha=0.5, label='Keras')

        sizes = [
            d.get('file_size_kb', d.get('model_size_kb', 0)) if k == 'keras'
            else d.get('model_size_kb', 0)
            for k, d in entries
        ]
        bars = ax_size.bar(x, sizes, color=colors, alpha=0.8, edgecolor='white')
        _keras_hline(ax_size, keras_size)
        for i, (bar, val) in enumerate(zip(bars, sizes)):
            top = bar.get_height() + max(sizes) * 0.01
            if i == 0:
                ann = f'{val:.0f} KB'
            else:
                diff = val - keras_size
                ann  = f'{val:.0f} KB\n({diff:+.0f} KB)'
            ax_size.text(bar.get_x() + bar.get_width() / 2, top,
                         ann, ha='center', va='bottom', fontsize=9)
        ax_size.set_xticks(x); ax_size.set_xticklabels(labels)
        ax_size.set_ylabel('Size (KB)'); ax_size.set_title('Model Size')
        ax_size.set_ylim(0, max(sizes) * 1.25); ax_size.grid(axis='y', alpha=0.3)

        ratios = [keras_size / s if s > 0 else 1.0 for s in sizes]
        bars   = ax_ratio.bar(x, ratios, color=colors, alpha=0.8, edgecolor='white')
        _keras_hline(ax_ratio, 1.0)
        for i, (bar, val) in enumerate(zip(bars, ratios)):
            ax_ratio.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.03,
                          f'{val:.2f}x' if i > 0 else '1.00x',
                          ha='center', va='bottom', fontsize=9)
        ax_ratio.set_xticks(x); ax_ratio.set_xticklabels(labels)
        ax_ratio.set_ylabel('Compression Ratio (x)'); ax_ratio.set_title('Compression Ratio')
        ax_ratio.set_ylim(0, max(ratios) * 1.2); ax_ratio.grid(axis='y', alpha=0.3)

        accuracies = [d.get('accuracy', 0) for _, d in entries]
        bars = ax_acc.bar(x, accuracies, color=colors, alpha=0.8, edgecolor='white')
        _keras_hline(ax_acc, keras_acc)
        for i, (bar, val) in enumerate(zip(bars, accuracies)):
            top = bar.get_height() + 0.002
            ann = (f'{val:.4f}' if i == 0
                   else f'{val:.4f}\n({val - keras_acc:+.4f})')
            ax_acc.text(bar.get_x() + bar.get_width() / 2, top,
                        ann, ha='center', va='bottom', fontsize=9)
        ax_acc.set_xticks(x); ax_acc.set_xticklabels(labels)
        ax_acc.set_ylabel('Accuracy'); ax_acc.set_title('Accuracy')
        ax_acc.set_ylim(0, min(1.0, max(accuracies) * 1.12)); ax_acc.grid(axis='y', alpha=0.3)

        has_times = any(d.get('mean_inference_time_ms') is not None for _, d in entries)
        if has_times:
            times   = [d.get('mean_inference_time_ms') or 0 for _, d in entries]
            keras_t = times[0]
            bars    = ax_time.bar(x, times, color=colors, alpha=0.8, edgecolor='white')
            _keras_hline(ax_time, keras_t)
            for i, (bar, val) in enumerate(zip(bars, times)):
                if val > 0:
                    top = bar.get_height() + max(times) * 0.01
                    ann = (f'{val:.1f}ms' if i == 0
                           else f'{val:.1f}ms\n({val - keras_t:+.1f}ms)')
                    ax_time.text(bar.get_x() + bar.get_width() / 2, top,
                                 ann, ha='center', va='bottom', fontsize=9)
            ax_time.set_xticks(x); ax_time.set_xticklabels(labels)
            ax_time.set_ylabel('Mean Inference Time (ms)'); ax_time.set_title('Inference Time')
            ax_time.set_ylim(0, max(times) * 1.3); ax_time.grid(axis='y', alpha=0.3)
        else:
            ax_time.text(0.5, 0.5,
                         'Inference time not available.\nCall benchmark_keras_inference() first.',
                         ha='center', va='center', transform=ax_time.transAxes,
                         fontsize=10, color='grey')
            ax_time.set_title('Inference Time'); ax_time.axis('off')

        plt.suptitle('Keras vs TFLite Performance Comparison')
        self._save_fig('full_comparison.png')

    def plot_per_class_f1_delta(self, figsize: Tuple[int, int] = (14, 5)) -> None:
        has_f1 = any('per_class_f1' in r for r in self.benchmark_results.values())
        if not self._guard(has_f1,
                           'No per-class F1 found. Call benchmark_all() first.'):
            return

        has_keras_f1 = (
            self.keras_model is not None
            and self.evaluation_handler is not None
            and self.evaluation_handler.report is not None
        )
        if not has_keras_f1:
            print('No Keras per-class F1 available. Call evaluation_handler.predict() first.')
            return

        report         = self.evaluation_handler.report
        keras_f1       = {
            label: report[label]['f1-score']
            for label in self.class_labels
            if label in report
        }
        tflite_types   = [mt for mt in self.benchmark_results
                          if 'per_class_f1' in self.benchmark_results[mt]]
        n_variants     = len(tflite_types)

        if n_variants == 0:
            print('No TFLite variants with per_class_f1.')
            return

        keras_arr = np.array([keras_f1.get(l, np.nan) for l in self.class_labels])

        delta_matrices = {}
        for mt in tflite_types:
            tflite_f1 = self.benchmark_results[mt]['per_class_f1']
            tflite_arr = np.array([tflite_f1.get(l, np.nan) for l in self.class_labels])
            delta_matrices[mt] = tflite_arr - keras_arr

        abs_max = max(
            np.nanmax(np.abs(dm)) for dm in delta_matrices.values()
        )
        abs_max = max(abs_max, 0.01)

        fig, axes = plt.subplots(1, n_variants, figsize=figsize, sharey=True)
        if n_variants == 1:
            axes = [axes]

        fig.suptitle(
            'Per-Class F1 Delta',
            fontsize=12, fontweight='bold',
        )

        for ax, mt in zip(axes, tflite_types):
            delta  = delta_matrices[mt]
            x      = np.arange(len(self.class_labels))
            colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in delta]

            bars = ax.barh(x, delta, color=colors, alpha=0.82, edgecolor='white')
            ax.axvline(0, color='black', linewidth=1.0, alpha=0.6)

            for bar, val, label in zip(bars, delta, self.class_labels):
                if not np.isnan(val):
                    ha  = 'left' if val >= 0 else 'right'
                    xpos = val + (abs_max * 0.02 if val >= 0 else -abs_max * 0.02)
                    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                            f'{val:+.3f}', va='center', ha=ha, fontsize=8)

            for i, (label, kf1) in enumerate(zip(self.class_labels, keras_arr)):
                ax.text(-abs_max * 1.05, i,
                        f'({kf1:.2f})', va='center', ha='right',
                        fontsize=7, color='#555555')

            ax.set_yticks(x)
            ax.set_yticklabels(self.class_labels, fontsize=9)
            ax.set_xlim(-abs_max * 1.4, abs_max * 1.4)
            ax.set_xlabel('Delta F1 Score')
            ax.set_title(self._LABEL_MAP.get(mt, mt), fontsize=11)
            ax.grid(axis='x', alpha=0.3)

        if n_variants > 1:
            axes[0].text(-abs_max * 1.05, -0.7,
                         'Keras', va='top', ha='right',
                         fontsize=7, color='#555555', style='italic')

        plt.tight_layout()
        self._save_fig('per_class_f1_delta.png')

    def generate_summary(self, mode: str) -> None:
        summary_data = [
            ('Model',    self.config.get('model', 'Unknown')),
            ('Strategy', self.config.get('strategy', 'Unknown')),
        ]

        if self.keras_model:
            summary_data += [
                None,
                ('Keras accuracy',  f"{self.keras_model['accuracy']:.4f}"),
                ('Keras loss',      f"{self.keras_model['test_loss']:.4f}" if self.keras_model.get('test_loss') is not None else 'n/a'),
                ('Keras params',    f"{self.keras_model['params']:,}"),
                ('Keras size (KB)', f"{self.keras_model.get('file_size_kb', self.keras_model['model_size_kb']):.2f}"),
            ]
            if self.keras_model.get('mean_inference_time_ms') is not None:
                summary_data += [
                    ('Keras mean inference', f"{self.keras_model['mean_inference_time_ms']:.2f} ms"),
                    ('Keras P95 inference',  f"{self.keras_model['p95_inference_time_ms']:.2f} ms"),
                    ('Keras samples tested', f"{self.keras_model['samples_tested']:,}"),
                ]

        keras_size      = self.keras_model.get('file_size_kb', self.keras_model['model_size_kb']) if self.keras_model else None
        keras_accuracy  = self.keras_model['accuracy']                   if self.keras_model else None
        keras_inference = self.keras_model.get('mean_inference_time_ms') if self.keras_model else None
        keras_p95       = self.keras_model.get('p95_inference_time_ms')  if self.keras_model else None

        for model_type, results in self.benchmark_results.items():
            label = self._LABEL_MAP.get(model_type, model_type)

            size_str    = f"{results['model_size_kb']:.2f}"
            compression = 'n/a'
            if keras_size is not None:
                size_diff   = results['model_size_kb'] - keras_size
                size_str    = f"{results['model_size_kb']:.2f} ({'+' if size_diff >= 0 else ''}{size_diff:.2f} KB)"
                compression = f"{results.get('compression_ratio', keras_size / results['model_size_kb']):.2f}x"

            model_accuracy = f"{results['accuracy']:.4f}"
            if keras_accuracy is not None:
                diff           = results['accuracy'] - keras_accuracy
                model_accuracy = f"{results['accuracy']:.4f} ({'+' if diff >= 0 else ''}{diff:.4f})"

            mean_ms  = results['mean_inference_time_ms']
            p95_ms   = results['p95_inference_time_ms']
            mean_str = f"{mean_ms:.2f} ms"
            p95_str  = f"{p95_ms:.2f} ms"

            if keras_inference is not None:
                inf_diff = mean_ms - keras_inference
                mean_str = f"{mean_ms:.2f} ms ({'+' if inf_diff >= 0 else ''}{inf_diff:.2f} ms)"
            if keras_p95 is not None:
                p95_diff = p95_ms - keras_p95
                p95_str  = f"{p95_ms:.2f} ms ({'+' if p95_diff >= 0 else ''}{p95_diff:.2f} ms)"

            summary_data += [
                None,
                (f'{label} accuracy',         model_accuracy),
                (f'{label} F1 macro',         f"{results.get('f1_macro', 0):.4f}"),
                (f'{label} size (KB)',        size_str),
                (f'{label} compression',      compression),
                (f'{label} mean inference',   mean_str),
                (f'{label} P95 inference',    p95_str),
                (f'{label} samples tested',   f"{results['samples_tested']:,}"),
                (f'{label} conversion time',  f"{self.conversion_times.get(model_type, 0):.1f}s"),
            ]

        if mode == 'latex':
            self._generate_latex_summary('TFLiteHandler', summary_data, 'tflite_summary.tex')
        else:
            self._generate_ascii_summary('TFLiteHandler', summary_data)