import os
import gc
import json
import time
import zipfile
import datetime
from typing import Optional

import tensorflow as tf

from handlers import DatasetHandler, DataAugmentationHandler


class DataExperimentOrchestrator():
    """
    Orchestrator resposible for handling whole experiment runthrough.
    """

    def __init__(self, config: dict):
        self._config          = config
        self._archive_directory: Optional[str] = None
        self._experiment_orchestrator = None
        self.timestamp_start = datetime.datetime.now()
        self.timestamp_stop: Optional[datetime.datetime] = None
        self.timestamp       = self.timestamp_start.strftime('%Y%m%d-%H%M%S')

        self.experiment_name = (
            f'{config["dataset"]}_'
            f'{config["augmentation"]['preset']}_'
            f'{self.timestamp}'
        )

        self._dataset_handler           = None
        self._data_augmentation_handler = None

        print('DataExperimentOrchestrator initialized.')
        print(f'Experiment name: {self.experiment_name}')
        print(f'Dataset:         {config["dataset"]}')
        print(f'Augmentation:    {config["augmentation"].get("preset", "Custom")} '
              f'(enabled={config["augmentation"].get("enabled", True)})')
        print(f'Start time:      {self.timestamp_start.strftime("%Y-%m-%d %H:%M:%S")}')

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

    # ── saving ────────────────────────────────────────────────

    def save_config(self, name: str = 'config') -> str:
        """Save CONFIG dictionary as JSON."""
        path = os.path.join(self.archive_directory, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
        print(f"Config saved to: {path}")
        return path

    # ── archive directory ────────────────────────────────────

    @property
    def archive_directory(self) -> Optional[str]:
        """Lazy lookup: own value first, then via ExperimentHandler back-reference."""
        if self._archive_directory is not None:
            return self._archive_directory
        if self._experiment_orchestrator is not None:
            return self._experiment_orchestrator.archive_directory
        return None

    @archive_directory.setter
    def archive_directory(self, value: str) -> None:
        self._archive_directory = value

    # ── archive ───────────────────────────────────────────────

    def archive_experiment(self) -> None:
        """
        Archive dataset + augmentation config only.
        Saves config JSON and LaTeX summaries for dataset and augmentation handlers.
        """
        self.timestamp_stop = datetime.datetime.now()

        print('\n' + '=' * 60)
        print('ARCHIVING DATA EXPERIMENT')
        print('=' * 60)
        print(f'  Start:   {self.timestamp_start.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'  Stop:    {self.timestamp_stop.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'  Elapsed: {self._fmt_duration(self.elapsed_seconds)}')
        print('=' * 60)

        self.save_config()
        self.save_latex_summaries()

        print('=' * 60)
        print(f'Data experiment archived to: {self.archive_directory}')
        print('=' * 60 + '\n')

    def save_latex_summaries(self) -> None:
        """Save LaTeX summaries for dataset and augmentation handlers only."""
        print('\nSaving LaTeX summaries...')
        for h in [self._dataset_handler, self._data_augmentation_handler]:
            if h is not None:
                try:
                    h.generate_summary(mode='latex')
                except NotImplementedError:
                    pass
                except Exception as e:
                    print(f'{h.__class__.__name__}.generate_summary() failed: {e}')

    def is_complete(self) -> bool:
        """Check if archive contains config.json (minimal requirement)."""
        return os.path.exists(os.path.join(self.archive_directory, 'config.json'))

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
            from google.colab import files
            print(f"Starting download: {os.path.basename(zip_path)}")
            files.download(zip_path)
        except ImportError:
            print(f"Not running in Colab. Zip available at: {zip_path}")

        return zip_path

    # ── summary ───────────────────────────────────────────────

    def _build_summary_dict(self) -> dict:
        """Structured summary — dataset and augmentation only."""
        summary = {
            'experiment': {
                'name':            self.experiment_name,
                'timestamp_start': self.timestamp_start.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_stop':  self.timestamp_stop.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp_stop else None,
                'elapsed':         self._fmt_duration(self.elapsed_seconds),
            },
            'config': dict(self._config),
            'dataset':      None,
            'augmentation': None,
        }

        if self._dataset_handler is not None:
            try:
                dh = self._dataset_handler
                summary['dataset'] = {
                    'name':          getattr(dh, 'dataset_name', self._config['dataset']),
                    'class_num':     getattr(dh, 'class_num',    None),
                    'class_labels':  getattr(dh, 'class_labels', None),
                    'train_samples': getattr(dh, 'train_samples', None),
                    'val_samples':   getattr(dh, 'val_samples',   None),
                    'test_samples':  getattr(dh, 'test_samples',  None),
                }
            except Exception as e:
                summary['dataset'] = {'error': str(e)}

        if self._data_augmentation_handler is not None:
            try:
                ah = self._data_augmentation_handler
                summary['augmentation'] = {
                    'enabled': self._config['augmentation'].get('enabled'),
                    'preset':  self._config['augmentation'].get('preset'),
                    'params':  {
                        k: v for k, v in self._config['augmentation'].items()
                        if k not in ('enabled', 'preset')
                    },
                }
            except Exception as e:
                summary['augmentation'] = {'error': str(e)}

        return summary

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
        lines.append('DATA EXPERIMENT SUMMARY REPORT')
        rule()
        exp = summary['experiment']
        lines.append(f"  Name      : {exp['name']}")
        lines.append(f"  Start     : {exp['timestamp_start']}")
        lines.append(f"  Stop      : {exp['timestamp_stop'] or 'n/a'}")
        lines.append(f"  Elapsed   : {exp['elapsed']}")

        section('DATASET')
        ds = summary.get('dataset')
        if ds is None:
            lines.append('  No DatasetHandler registered.')
        elif 'error' in ds:
            lines.append(f"  Error: {ds['error']}")
        else:
            row('Name',          str(ds.get('name')))
            row('Classes',       str(ds.get('class_num')))
            row('Labels',        str(ds.get('class_labels')))
            row('Train samples', str(ds.get('train_samples')))
            row('Val samples',   str(ds.get('val_samples')))
            row('Test samples',  str(ds.get('test_samples')))

        section('AUGMENTATION')
        aug = summary.get('augmentation')
        if aug is None:
            lines.append('  No DataAugmentationHandler registered.')
        elif 'error' in aug:
            lines.append(f"  Error: {aug['error']}")
        else:
            row('Enabled', str(aug.get('enabled')))
            row('Preset',  str(aug.get('preset')))
            for param, val in (aug.get('params') or {}).items():
                row(param, str(val))

        lines.append('')
        rule()
        return lines

    # ── shutdown ───────────────────────────────────────

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
            from google.colab import runtime
            print(f"Shutting down runtime in {delay}s...")
            time.sleep(delay)
            runtime.unassign()
        except ImportError:
            print("Not running in Colab — runtime shutdown skipped.")