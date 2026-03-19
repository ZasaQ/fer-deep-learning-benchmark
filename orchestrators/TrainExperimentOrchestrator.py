import os
import gc
import json
import pickle
import time
import zipfile
import datetime
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from google.colab import files

from handlers import (
    DatasetHandler,
    DataAugmentationHandler,
    CallbacksHandler,
    ModelHandler,
    TrainingHandler,
    EvaluationHandler,
    TFLiteHandler,
)
from DirectoryManager import DirectoryManager


class TrainExperimentOrchestrator:
    """
    Manages experiment metadata, archiving, summary generation,
    and experiment download.
    """

    def __init__(self, config: dict):
        self.config = config
        self.timestamp_start = datetime.datetime.now()
        self.timestamp_stop: Optional[datetime.datetime] = None
        self.timestamp = self.timestamp_start.strftime('%Y%m%d-%H%M%S')
        self.experiment_name = (
            f'{config["id"]}_'
            f'{config["dataset"]}_'
            f'{config["model"]}_'
            f'{config["strategy"]}_'
            f'{self.timestamp}'
        )
        self.model_name = (
            f'{config["dataset"]}_'
            f'{config["model"]}_'
            f'{config["strategy"]}_'
            f'{self.timestamp}'
        )

        self._dataset_handler           = None
        self._data_augmentation_handler = None
        self._callbacks_handler         = None
        self._model_handler             = None
        self._training_handler          = None
        self._evaluation_handler        = None
        self._tflite_handler            = None

        print('TrainExperimentOrchestrator has been initialized.')
        print(f'Experiment name: {self.experiment_name}')
        print(f'Model name:      {self.model_name}')
        print(f'Start time:      {self.timestamp_start.strftime("%Y-%m-%d %H:%M:%S")}')

    # ── registration ─────────────────────────────────────────

    def register_dataset(self, dataset_handler: DatasetHandler) -> None:
        """Register DatasetHandler."""
        self._dataset_handler = dataset_handler
        dataset_handler._experiment_orchestrator = self
        print('DatasetHandler registered.')

    def register_data_augmentation(self, data_augmentation_handler: DataAugmentationHandler) -> None:
        """Register DataAugmentationHandler."""
        self._data_augmentation_handler = data_augmentation_handler
        data_augmentation_handler._experiment_orchestrator = self
        print('DataAugmentationHandler registered.')

    def register_callbacks(self, callbacks_handler: CallbacksHandler) -> None:
        """Register CallbacksHandler."""
        self._callbacks_handler = callbacks_handler
        callbacks_handler._experiment_orchestrator = self
        print('CallbacksHandler registered.')

    def register_model(self, model_handler: ModelHandler) -> None:
        """Register ModelHandler."""
        self._model_handler = model_handler
        model_handler._experiment_orchestrator = self
        print('ModelHandler registered.')

    def register_training(self, training_handler: TrainingHandler) -> None:
        """Register TrainingHandler."""
        self._training_handler = training_handler
        training_handler._experiment_orchestrator = self
        print('TrainingHandler registered.')

    def register_evaluation(self, evaluation_handler: EvaluationHandler) -> None:
        """Register EvaluationHandler."""
        self._evaluation_handler = evaluation_handler
        evaluation_handler._experiment_orchestrator = self
        print('EvaluationHandler registered.')

    def register_tflite(self, tflite_handler: TFLiteHandler) -> None:
        """Register TFLiteHandler."""
        self._tflite_handler = tflite_handler
        tflite_handler._experiment_orchestrator = self
        print('TFLiteHandler registered.')

    # ── configuration ────────────────────────────────────────

    def configure_archive(self, dir_handler: DirectoryManager) -> None:
        """Set archive_directory on self."""
        self.archive_directory = dir_handler.get('archive')
        print(f'Archive directory configured: {self.archive_directory}')

    # ── timing ───────────────────────────────────────────────

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Elapsed seconds between start and stop (or now if not yet stopped)."""
        end = self.timestamp_stop or datetime.datetime.now()
        return (end - self.timestamp_start).total_seconds()

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

    # ── saving ───────────────────────────────────────────────

    def save_history(self) -> str:
        """Save training history as pickle. Requires register_training()."""
        if self._training_handler is None:
            print("No TrainingHandler registered (skipping history).")
            return ''

        path = os.path.join(self.archive_directory, 'history.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self._training_handler.history.history, f)
        print(f"History saved to: {path}")
        return path

    def save_predictions(self) -> str:
        """Save evaluation predictions as .npz. Requires register_evaluation()."""
        if self._evaluation_handler is None:
            print("No EvaluationHandler registered (skipping predictions).")
            return ''

        ev = self._evaluation_handler
        if ev.y_true is None:
            print("No predictions available (skipping).")
            return ''

        path = os.path.join(self.archive_directory, 'predictions.npz')
        np.savez(path, y_true=ev.y_true, y_pred=ev.y_pred, y_pred_proba=ev.y_pred_proba)
        print(f"Predictions saved to: {path}")
        return path

    def save_config(self) -> str:
        """Save CONFIG dictionary as JSON."""
        config_filename = f"config_{self.experiment_name}.json"
        path            = os.path.join(self.archive_directory, config_filename)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to:  {path}")
        return path

    def save_benchmark_results(self) -> str:
        """Save TFLite benchmark results as JSON. Requires register_tflite()."""
        if self._tflite_handler is None:
            print("No TFLiteHandler registered (skipping benchmark results).")
            return ''

        if not self._tflite_handler.benchmark_results:
            print("No benchmark results available (skipping).")
            return ''

        serializable = {}
        for model_type, results in self._tflite_handler.benchmark_results.items():
            serializable[model_type] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results.items()
                if k != 'raw_inference_times_ms'
            }

        path = os.path.join(self.archive_directory, 'benchmark_results.json')
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Benchmark results saved to: {path}")
        return path

    def save_metrics_json(self) -> str:
        """
        Save a flat metrics JSON.
        Combines evaluation metrics, training summary, TFLite benchmarks,
        ROC AUC per class, and experiment metadata into a single file.
        """
        if self._evaluation_handler is None:
            print("No EvaluationHandler registered (skipping metrics).")
            return ''

        ev = self._evaluation_handler

        if ev.y_true is None or ev.y_pred is None:
            print("No predictions available — call predict() before archiving.")
            return ''

        # classification report
        report   = ev._get_classification_report()
        weighted = report.get('weighted avg', {})
        macro    = report.get('macro avg', {})
        cm       = np.array(ev.confusion_matrix)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        # ROC AUC per class
        roc_auc_per_class = {}
        macro_auc = None
        if ev.y_pred_proba is not None:
            try:
                n_classes  = ev.dataset_handler.class_num
                y_true_bin = label_binarize(ev.y_true, classes=range(n_classes))

                for i, label in enumerate(ev.dataset_handler.class_labels):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], ev.y_pred_proba[:, i])
                    roc_auc_per_class[label] = round(float(auc(fpr, tpr)), 6)

                macro_auc = round(float(
                    roc_auc_score(y_true_bin, ev.y_pred_proba, average='macro')
                ), 6)
            except Exception as e:
                print(f"ROC AUC computation failed: {e}")

        # training info
        training_info = {}
        if self._training_handler is not None:
            try:
                history  = self._training_handler.history.history
                val_acc  = history.get('val_accuracy', history.get('val_acc', []))
                val_loss = history.get('val_loss', [])
                best_epoch = int(np.argmax(val_acc)) if val_acc else None
                training_info = {
                    'actual_epochs':     len(val_acc),
                    'best_epoch':        (best_epoch + 1) if best_epoch is not None else None,
                    'best_val_accuracy': round(float(max(val_acc)), 6) if val_acc else None,
                    'best_val_loss':     round(float(val_loss[best_epoch]), 6)
                                         if val_loss and best_epoch is not None else None,
                    'early_stopping_triggered': (
                        len(val_acc) < self.config.get('epochs', len(val_acc))
                        if val_acc else None
                    ),
                }
            except Exception as e:
                training_info = {'error': str(e)}

        # TFLite summary
        tflite_info = {}
        if self._tflite_handler is not None and self._tflite_handler.benchmark_results:
            try:
                for model_type, results in self._tflite_handler.benchmark_results.items():
                    tflite_info[model_type] = {
                        'accuracy':        round(float(results['accuracy']), 6),
                        'size_kb':         round(float(results['model_size_kb']), 2),
                        'mean_latency_ms': round(float(results['mean_inference_time_ms']), 4),
                        'p95_latency_ms':  round(float(results['p95_inference_time_ms']), 4),
                    }
            except Exception as e:
                tflite_info = {'error': str(e)}

        metrics = {
            'experiment_id':   self.experiment_name,
            'model':           self.config.get('model'),
            'dataset':         self.config.get('dataset'),
            'strategy':        self.config.get('strategy'),
            'timestamp_start': self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_stop':  self.timestamp_stop.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp_stop else None,
            'elapsed_seconds': round(self.elapsed_seconds, 1),

            'accuracy':        round(float(ev.test_accuracy), 6) if ev.test_accuracy is not None else None,
            'test_loss':       round(float(ev.test_loss), 6)     if ev.test_loss     is not None else None,
            'f1_macro':        round(float(macro.get('f1-score', 0)), 6),
            'f1_weighted':     round(float(weighted.get('f1-score', 0)), 6),
            'precision_macro': round(float(macro.get('precision', 0)), 6),
            'recall_macro':    round(float(macro.get('recall', 0)), 6),
            'macro_auc':       macro_auc,

            'per_class_f1': {
                label: round(float(report[label]['f1-score']), 6)
                for label in ev.dataset_handler.class_labels
            },
            'per_class_accuracy': {
                label: round(float(acc), 6)
                for label, acc in zip(ev.dataset_handler.class_labels, per_class_acc)
            },
            'roc_auc_per_class': roc_auc_per_class,

            'confusion_matrix': cm.tolist(),
            'class_names':      ev.dataset_handler.class_labels,
            'n_test_samples':   int(len(ev.y_true)),

            'training': training_info,

            'tflite': tflite_info,
        }

        path = os.path.join(self.archive_directory, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to: {path}")
        return path

    def save_latex_summaries(self) -> None:
        """Save LaTeX summary .tex files for all registered handlers."""
        print("\nSaving LaTeX summaries...")
        for h in [
            self._dataset_handler,
            self._data_augmentation_handler,
            self._callbacks_handler,
            self._model_handler,
            self._training_handler,
            self._evaluation_handler,
            self._tflite_handler,
        ]:
            if h is not None:
                try:
                    h.generate_summary(mode='latex')
                except NotImplementedError:
                    pass
                except Exception as e:
                    print(f"{h.__class__.__name__}.generate_summary() failed: {e}")
            else:
                print('No LaTeX summaries to be saved')

    # ── archiving ────────────────────────────────────────────

    def archive_experiment(self) -> None:
        """
        Archive all experiment data: config, history, predictions,
        benchmark results, metrics JSON, and LaTeX summary tables.

        Note:
            Keras and TFLite model files are NOT moved - they remain in their
            original directories managed by CallbackHandler / TFLiteHandler.
        """
        self.timestamp_stop = datetime.datetime.now()

        print("\n" + "=" * 60)
        print("ARCHIVING EXPERIMENT")
        print("=" * 60)
        print(f"  Start:   {self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Stop:    {self.timestamp_stop.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Elapsed: {self._fmt_duration(self.elapsed_seconds)}")
        print("=" * 60)

        self.save_history()
        self.save_predictions()
        self.save_benchmark_results()
        self.save_metrics_json()
        self.save_latex_summaries()

        print("=" * 60)
        print(f"Experiment archived to: {self.archive_directory}")
        print("=" * 60 + "\n")

    # ── zip & download ───────────────────────────────────────

    def create_zip(self, output_path: Optional[str] = None) -> str:
        """Zip the entire experiment root folder (all subdirectories included)."""
        root = self.experiment_name

        if output_path is None:
            parent = os.path.dirname(os.path.abspath(root))
            output_path = os.path.join(parent, f'{self.experiment_name}.zip')

        print(f"Creating zip archive of: {root}")

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _, files in os.walk(root):
                for filename in files:
                    if filename.endswith('.zip'):
                        continue
                    filepath = os.path.join(dirpath, filename)
                    arcname = os.path.relpath(filepath, os.path.dirname(root))
                    zf.write(filepath, arcname)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Zip created: {output_path}  ({size_mb:.2f} MB)")
        return output_path

    def download_experiment(self) -> str:
        """
        Zip the entire experiment root folder and trigger a browser download (Colab).

        Returns:
            Path to the created zip file
        """
        zip_path = self.create_zip()

        try:
            print(f"Starting download: {os.path.basename(zip_path)}")
            files.download(zip_path)
        except ImportError:
            print(f"Not running in Colab. Zip available at: {zip_path}")

        return zip_path

    # ── utilities ────────────────────────────────────────────

    def list_contents(self) -> None:
        """List all files in the archive directory."""
        if not os.path.exists(self.archive_directory):
            print("No archive directory found.")
            return

        files = os.listdir(self.archive_directory)
        if not files:
            print("Archive is empty.")
            return

        print(f"\nArchive contents: {self.archive_directory}")
        print("-" * 60)

        total_mb = 0.0
        for f in sorted(files):
            size = os.path.getsize(os.path.join(self.archive_directory, f))
            size_mb = size / (1024 * 1024)
            total_mb += size_mb
            print(f"  {f:<40} {size_mb:>8.2f} MB")

        print("-" * 60)
        print(f"  {'TOTAL':<40} {total_mb:>8.2f} MB\n")

    def is_complete(self) -> bool:
        """Check if archive contains all required files."""
        required = ['history.pkl', 'config.json', 'metrics.json']
        return all(
            os.path.exists(os.path.join(self.archive_directory, f))
            for f in required
        )

    # ── summary building ─────────────────────────────────────

    def _build_summary_dict(self) -> dict:
        """Collect all available data into a structured summary dictionary."""
        summary = {
            'experiment': {
                'name':            self.experiment_name,
                'timestamp_start': self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_stop':  self.timestamp_stop.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp_stop else None,
                'elapsed':         self._fmt_duration(self.elapsed_seconds),
            },
            'config': {k: v for k, v in self.config.items()},
            'training': None,
            'evaluation': None,
            'tflite': None,
        }

        if self._training_handler is not None:
            try:
                history    = self._training_handler.history.history
                val_acc    = history.get('val_accuracy', history.get('val_acc', []))
                val_loss   = history.get('val_loss', [])
                train_acc  = history.get('accuracy', history.get('acc', []))
                train_loss = history.get('loss', [])

                best_epoch   = int(np.argmax(val_acc)) + 1 if val_acc else None
                total_epochs = len(val_acc) if val_acc else None

                summary['training'] = {
                    'total_epochs_run':         total_epochs,
                    'best_epoch':               best_epoch,
                    'best_val_accuracy':        round(float(max(val_acc)), 6) if val_acc else None,
                    'best_val_loss':            round(float(val_loss[best_epoch - 1]), 6) if val_loss and best_epoch else None,
                    'final_train_accuracy':     round(float(train_acc[-1]), 6) if train_acc else None,
                    'final_train_loss':         round(float(train_loss[-1]), 6) if train_loss else None,
                    'early_stopping_triggered': (
                        total_epochs < self.config.get('epochs', total_epochs)
                        if total_epochs is not None else None
                    ),
                }
            except Exception as e:
                summary['training'] = {'error': str(e)}

        if self._evaluation_handler is not None:
            try:
                ev        = self._evaluation_handler
                eval_data = {}

                if hasattr(ev, 'test_accuracy') and ev.test_accuracy is not None:
                    eval_data['accuracy'] = round(float(ev.test_accuracy), 6)

                if hasattr(ev, 'report') and ev.report is not None:
                    report = ev.report
                    if isinstance(report, dict):
                        weighted = report.get('weighted avg', {})
                        macro    = report.get('macro avg', {})
                        eval_data['f1_weighted']        = round(float(weighted.get('f1-score', 0)), 6)
                        eval_data['f1_macro']           = round(float(macro.get('f1-score', 0)), 6)
                        eval_data['precision_weighted'] = round(float(weighted.get('precision', 0)), 6)
                        eval_data['recall_weighted']    = round(float(weighted.get('recall', 0)), 6)

                if hasattr(ev, 'confusion_matrix') and ev.confusion_matrix is not None:
                    cm            = np.array(ev.confusion_matrix)
                    per_class_acc = cm.diagonal() / cm.sum(axis=1)
                    eval_data['confusion_matrix_summary'] = {
                        'per_class_accuracy': {
                            str(i): round(float(v), 4)
                            for i, v in enumerate(per_class_acc)
                        },
                        'worst_class': int(np.argmin(per_class_acc)),
                        'best_class':  int(np.argmax(per_class_acc)),
                    }

                summary['evaluation'] = eval_data

            except Exception as e:
                summary['evaluation'] = {'error': str(e)}

        if self._tflite_handler is not None:
            try:
                th          = self._tflite_handler
                tflite_data = {}

                keras_size = None
                if hasattr(th, 'keras_model') and th.keras_model is not None:
                    keras_size = th.keras_model.get('model_size_kb')
                    tflite_data['keras_model'] = {
                        k: v for k, v in th.keras_model.items()
                        if k != 'raw_inference_times_ms'
                    }

                float32_size  = th.model_sizes.get('float32')
                baseline_size = (keras_size * 1024) if keras_size else float32_size

                for model_type, results in th.benchmark_results.items():
                    size_kb     = results['model_size_kb']
                    compression = (
                        round(baseline_size / (size_kb * 1024), 4)
                        if baseline_size else None
                    )
                    tflite_data[model_type] = {
                        'accuracy':                      round(float(results['accuracy']), 6),
                        'model_size_kb':                 round(float(size_kb), 2),
                        'compression_ratio_vs_baseline': compression,
                        'mean_inference_time_ms':        round(float(results['mean_inference_time_ms']), 4),
                        'std_inference_time_ms':         round(float(results['std_inference_time_ms']), 4),
                        'p95_inference_time_ms':         round(float(results['p95_inference_time_ms']), 4),
                        'samples_tested':                results['samples_tested'],
                    }

                summary['tflite'] = tflite_data

            except Exception as e:
                summary['tflite'] = {'error': str(e)}

        return summary

    # ── txt formatting ───────────────────────────────────────

    def _format_txt(self, summary: dict) -> list:
        """Format summary dict into human-readable lines."""
        W     = 64
        lines = []

        def rule(char='='):
            lines.append(char * W)

        def section(title):
            lines.append('')
            lines.append(title)
            rule('-')

        def row(label, value, indent=2):
            lines.append(f"{'  ' * (indent // 2)}{label:<36} {value}")

        rule()
        lines.append('EXPERIMENT SUMMARY REPORT')
        rule()
        exp = summary['experiment']
        lines.append(f"  Name      : {exp['name']}")
        lines.append(f"  Start     : {exp['timestamp_start']}")
        lines.append(f"  Stop      : {exp['timestamp_stop'] or 'n/a'}")
        lines.append(f"  Elapsed   : {exp['elapsed']}")

        section('CONFIGURATION')
        for key, value in summary['config'].items():
            row(key, str(value))

        if summary['training'] is not None:
            section('TRAINING')
            tr = summary['training']
            if 'error' in tr:
                lines.append(f"  Error: {tr['error']}")
            else:
                row('Total epochs run',         str(tr.get('total_epochs_run')))
                row('Best epoch',               str(tr.get('best_epoch')))
                row('Best val accuracy',        str(tr.get('best_val_accuracy')))
                row('Best val loss',            str(tr.get('best_val_loss')))
                row('Final train accuracy',     str(tr.get('final_train_accuracy')))
                row('Final train loss',         str(tr.get('final_train_loss')))
                row('Early stopping triggered', str(tr.get('early_stopping_triggered')))

        if summary['evaluation'] is not None:
            section('EVALUATION')
            ev = summary['evaluation']
            if 'error' in ev:
                lines.append(f"  Error: {ev['error']}")
            else:
                row('Accuracy',             str(ev.get('accuracy')))
                row('F1 (weighted)',        str(ev.get('f1_weighted')))
                row('F1 (macro)',           str(ev.get('f1_macro')))
                row('Precision (weighted)', str(ev.get('precision_weighted')))
                row('Recall (weighted)',    str(ev.get('recall_weighted')))

                cm_summary = ev.get('confusion_matrix_summary')
                if cm_summary:
                    lines.append('')
                    lines.append('  Per-class accuracy:')
                    for cls, acc in cm_summary['per_class_accuracy'].items():
                        lines.append(f"    Class {cls:<6} {acc:.4f}")
                    row('Best class',  str(cm_summary.get('best_class')))
                    row('Worst class', str(cm_summary.get('worst_class')))

        if summary['tflite'] is not None:
            section('TFLITE BENCHMARKS')
            tfl = summary['tflite']
            if 'error' in tfl:
                lines.append(f"  Error: {tfl['error']}")
            else:
                if 'keras_model' in tfl:
                    lines.append('')
                    lines.append('  [KERAS BASELINE]')
                    km = tfl['keras_model']
                    row('Accuracy',                    str(km.get('accuracy')),      indent=4)
                    row('Params',                      str(km.get('params')),         indent=4)
                    row('Size - params (KB)',           str(km.get('model_size_kb')), indent=4)
                    if 'file_size_kb' in km:
                        row('Size - file (KB)',         str(km.get('file_size_kb')), indent=4)
                    if km.get('mean_inference_time_ms') is not None:
                        row('Mean inference time (ms)', str(km.get('mean_inference_time_ms')), indent=4)
                        row('P95 inference time (ms)',  str(km.get('p95_inference_time_ms')),  indent=4)

                for model_type, data in tfl.items():
                    if model_type == 'keras_model':
                        continue
                    lines.append('')
                    lines.append(f"  [{model_type.upper()}]")
                    row('Accuracy',                      str(data.get('accuracy')),                       indent=4)
                    row('Model size (KB)',               str(data.get('model_size_kb')),                  indent=4)
                    row('Compression ratio vs baseline', str(data.get('compression_ratio_vs_baseline')),  indent=4)
                    row('Mean inference time (ms)',      str(data.get('mean_inference_time_ms')),         indent=4)
                    row('Std inference time (ms)',       str(data.get('std_inference_time_ms')),          indent=4)
                    row('P95 inference time (ms)',       str(data.get('p95_inference_time_ms')),          indent=4)
                    row('Samples tested',                str(data.get('samples_tested')),                 indent=4)

        lines.append('')
        rule()
        return lines

    # ── shutdown ─────────────────────────────────────────────

    def shutdown(self, delay: int) -> None:
        """
        Clear TensorFlow session, free memory, and disconnect Colab runtime.
        """
        print("Clearing TensorFlow session...")
        try:
            tf.keras.backend.clear_session()
            print("   TensorFlow session cleared.")
        except Exception as e:
            print(f"  TensorFlow clear failed: {e}")

        print("Running garbage collector...")
        gc.collect()
        print("Memory freed.")

        try:
            print(f"Shutting down runtime in {delay}s...")
            time.sleep(delay)
            runtime.unassign()
        except ImportError:
            print("Not running in Colab — runtime shutdown skipped.")