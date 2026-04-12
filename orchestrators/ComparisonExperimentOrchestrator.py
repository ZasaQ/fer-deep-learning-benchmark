import json
import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path
from datetime import datetime as _dt
from typing import Optional

from google.colab import drive, files


class ComparisonExperimentOrchestrator:

    DRIVE_MOUNT_PATH   = '/content/drive'
    DRIVE_SOURCE_PATH  = 'MyDrive/WSEI/R2S1/Magisterka/Trained'

    def __init__(self):
        self.timestamp       = _dt.now().strftime('%Y%m%d-%H%M%S')
        self.experiment_name = f'comparison_experiment_{self.timestamp}'

        self._comparison_experiment_dir = None
        self._archive_dir               = None

        self.trained_zips_dir     = None
        self.trained_unpacked_dir = None

        self._keras_handler  = None
        self._tflite_handler = None

        print(f'Experiment name: {self.experiment_name}')
        print('ComparisonExperimentOrchestrator initialized.')

    # ── registration ──────────────────────────────────────────────────────────

    def register_keras_handler(self, handler) -> None:
        """Register ComparisonKerasHandler."""
        self._keras_handler = handler
        print('ComparisonKerasHandler registered.')

    def register_tflite_handler(self, handler) -> None:
        """Register ComparisonTFLiteHandler."""
        self._tflite_handler = handler
        print('ComparisonTFLiteHandler registered.')

    # ── property ──────────────────────────────────────────────────────────────

    @property
    def comparison_experiment_dir(self) -> Optional[str]:
        return self._comparison_experiment_dir

    @comparison_experiment_dir.setter
    def comparison_experiment_dir(self, path: str) -> None:
        """Derive archive_dir from comparison_experiment_dir on assignment."""
        self._comparison_experiment_dir = path
        self._archive_dir               = os.path.join(path, 'archive')

    # ── drive ─────────────────────────────────────────────────────────────────

    def mount_drive(self) -> None:
        drive.mount(self.DRIVE_MOUNT_PATH)
        print(f'Google Drive mounted at {self.DRIVE_MOUNT_PATH}')

    def _drive_source_dir(self) -> str:
        return os.path.join(self.DRIVE_MOUNT_PATH, self.DRIVE_SOURCE_PATH)

    # ── copy ──────────────────────────────────────────────────────────────────

    def copy_experiments(self, pattern: Optional[str] = None) -> list:
        """Copies experiment ZIPs from Drive into local zips_dir. Skips existing."""
        self._check('copy_experiments', self.trained_zips_dir, 'zips_dir')

        source_dir = self._drive_source_dir()
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f'Drive source directory not found: {source_dir}. Call mount_drive() first.')

        os.makedirs(self.trained_zips_dir, exist_ok=True)

        available = sorted([f for f in os.listdir(source_dir) if f.endswith('.zip')])
        if pattern:
            available = [f for f in available if pattern in f]

        print(f'Copying {len(available)} ZIPs from Drive -> {self.trained_zips_dir}')
        copied, skipped = [], []
        for filename in available:
            src, dest = os.path.join(source_dir, filename), os.path.join(self.trained_zips_dir, filename)
            if os.path.exists(dest):
                skipped.append(dest); continue
            shutil.copy2(src, dest)
            copied.append(dest)
        return copied + skipped

    def unzip_experiments(self) -> list:
        """Unzips all experiment ZIPs from the local zips_dir into the local unpacked_dir."""
        self._check('unzip_experiments', self.trained_zips_dir,     'zips_dir')
        self._check('unzip_experiments', self.trained_unpacked_dir, 'unpacked_dir')
        
        zips = sorted([f for f in os.listdir(self.trained_zips_dir) if f.endswith('.zip')])
        
        unpacked = []
        for zip_name in zips:
            zip_path = os.path.join(self.trained_zips_dir, zip_name)
            dest = os.path.join(self.trained_unpacked_dir, os.path.splitext(zip_name)[0])
            if os.path.exists(dest):
                unpacked.append(dest); continue
            os.makedirs(dest)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            unpacked.append(dest)
        return unpacked

    # ── csv & summary ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_metrics(json_path: str) -> dict:
        with open(json_path, 'r') as f:
            d = json.load(f)
        tflite = d.get('tflite', {})
        float32, dynamic, int8 = tflite.get('float32', {}), tflite.get('dynamic_quant', {}), tflite.get('int8_quant', {})
        per_class_f1 = d.get('per_class_f1', {})
        emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        
        return {
            'id': d.get('experiment_name', ''), 'model': d.get('model_name', ''),
            'dataset': d.get('dataset', ''), 'strategy': d.get('strategy', ''),
            'f1_macro': d.get('f1_macro'), 'accuracy': d.get('test_accuracy'),
            **{f'f1_{e.lower()}': per_class_f1.get(e) for e in emotions},
            'tflite_float32_acc': float32.get('accuracy'), 'tflite_int8_acc': int8.get('accuracy'),
            'tflite_int8_size_kb': int8.get('model_size_kb'), 'tflite_int8_ms': int8.get('mean_inference_time_ms'),
        }

    def build_summary_csv(self, output_filename: str = 'summary_all_experiments.csv') -> Optional[str]:
        """Tworzy CSV na podstawie odpakowanych eksperymentów."""
        self._check('build_summary_csv', self.trained_unpacked_dir, 'unpacked_dir')
        self._check('build_summary_csv', self.comparison_experiment_dir, 'comparison_experiment_dir')

        experiments_dir = Path(self.trained_unpacked_dir)
        output_path = os.path.join(self.comparison_experiment_dir, output_filename)
        rows = []

        print(f"Generowanie podsumowania CSV: {output_filename}...")
        for exp_folder in sorted(experiments_dir.iterdir()):
            if not exp_folder.is_dir(): continue
            inner_folders = [f for f in exp_folder.iterdir() if f.is_dir()]
            if not inner_folders: continue
            
            archive = inner_folders[0] / 'archive'
            metrics_files = list(archive.glob('metrics_*.json')) if archive.exists() else []
            
            if metrics_files:
                try:
                    rows.append(self._extract_metrics(str(metrics_files[0])))
                except Exception as e: print(f'  [ERROR] {exp_folder.name}: {e}')

        if rows:
            df = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
            df.to_csv(output_path, index=False, float_format='%.4f')
            print(f'  [OK] Zapisano {len(df)} rekordów w {output_path}')
            return output_path
        print("  [!] Brak danych do CSV.")
        return None

    # ── archive & download ────────────────────────────────────────────────────

    def archive_results(self, archive_name: Optional[str] = None) -> str:
        """Creates a ZIP archive of the comparison experiment results, including the summary CSV."""
        self._check('archive_results', self._archive_dir, 'comparison_experiment_dir')

        self.build_summary_csv()

        os.makedirs(self._archive_dir, exist_ok=True)

        archive_name = archive_name or f'comparison_results_{self.timestamp}'
        archive_path = os.path.join(self._archive_dir, f'{archive_name}.zip')

        collected = 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for dirpath, _, filenames in os.walk(self._comparison_experiment_dir):
                for filename in filenames:
                    if filename.endswith('.zip'): 
                        continue

                    filepath = os.path.join(dirpath, filename)
                    arcname = os.path.relpath(filepath, os.path.dirname(self._comparison_experiment_dir))
                    zout.write(filepath, arcname)
                    collected += 1

        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f'Archived {collected} files -> {archive_path}, ({size_mb:.2f} MB)')
        return archive_path

    def download_archive(self, archive_path: Optional[str] = None) -> None:
        """Downloads the most recent (or specified) archive to local via Colab."""
        self._check('download_archive', self._archive_dir, 'archive_dir')
        if archive_path is None:
            zips = sorted([f for f in os.listdir(self._archive_dir) if f.endswith('.zip')],
                          key=lambda f: os.path.getmtime(os.path.join(self._archive_dir, f)))
            if not zips: raise FileNotFoundError("Brak archiwów do pobrania.")
            archive_path = os.path.join(self._archive_dir, zips[-1])

        print(f'Downloading: {os.path.basename(archive_path)}...')
        files.download(archive_path)

    @staticmethod
    def _check(method: str, value, attr: str):
        if value is None: raise RuntimeError(f'{method}() wymaga ustawienia {attr}.')