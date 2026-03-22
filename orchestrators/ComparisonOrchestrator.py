import os
import zipfile
from datetime import datetime as _dt
from typing import Optional

import gdown
from google.colab import files


class ComparisonOrchestrator:
    """Handles Google Drive download, ZIP unpacking, plot archiving and local download."""

    def __init__(self):
        self.timestamp       = _dt.now().strftime('%Y%m%d-%H%M%S')
        self.experiment_name = f'comparison_experiment_{self.timestamp}'

        self.zips_dir     = None
        self.unpacked_dir = None

        self._comparison_experiment_dir = None
        self._archive_dir               = None

        self._keras_handler  = None
        self._tflite_handler = None

        print(f'Experiment name: {self.experiment_name}')
        print('ComparisonOrchestrator initialized.')

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

    # ── validation ────────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        """Check if all required directories and handlers are configured."""
        checks = {
            'zips_dir':                  self.zips_dir is not None,
            'unpacked_dir':              self.unpacked_dir is not None,
            'comparison_experiment_dir': self._archive_dir is not None,
            'keras_handler':             self._keras_handler is not None,
            'tflite_handler':            self._tflite_handler is not None,
        }
        all_ok = all(checks.values())
        print('\nComparisonOrchestrator status:')
        for name, ok in checks.items():
            status = 'Ok' if ok else 'Error'
            print(f'  {status} {name}')
        return all_ok

    # ── download ──────────────────────────────────────────────────────────────

    def download_experiments(self, source: str) -> list:
        """Downloads experiment ZIPs from a Google Drive folder share link."""
        self._check('download_experiments', self.zips_dir, 'zips_dir')
        print(f'Downloading experiments from Google Drive -> {self.zips_dir}')
        gdown.download_folder(
            url=source,
            output=self.zips_dir,
            quiet=False,
            use_cookies=False,
        )
        zips = sorted([f for f in os.listdir(self.zips_dir) if f.endswith('.zip')])
        print(f'  Download complete — {len(zips)} ZIPs in {self.zips_dir}')
        return zips

    # ── unzip ─────────────────────────────────────────────────────────────────

    def unzip_experiments(self) -> list:
        """Unpacks ZIPs from zips/ into unpacked/, one folder per experiment. Skips existing."""
        self._check('unzip_experiments', self.zips_dir,     'zips_dir')
        self._check('unzip_experiments', self.unpacked_dir, 'unpacked_dir')
        zips = sorted([f for f in os.listdir(self.zips_dir) if f.endswith('.zip')])
        if not zips:
            raise FileNotFoundError(
                f'No ZIPs found in {self.zips_dir}. '
                'Call download_experiments() first.'
            )
        print(f'Unpacking {len(zips)} experiment ZIPs ...')
        unpacked = []
        for zip_name in zips:
            zip_path = os.path.join(self.zips_dir, zip_name)
            dest     = os.path.join(self.unpacked_dir, os.path.splitext(zip_name)[0])
            if os.path.exists(dest):
                print(f'  [skip] {zip_name} (already unpacked)')
                unpacked.append(dest)
                continue
            os.makedirs(dest)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            print(f'  {zip_name} -> {os.path.basename(dest)}/')
            unpacked.append(dest)
        print(f'  {len(unpacked)} experiments ready in {self.unpacked_dir}')
        return unpacked

    # ── archive ───────────────────────────────────────────────────────────────

    def archive_results(self, archive_name: Optional[str] = None) -> str:
        """Zips the entire comparison experiment folder into archive/."""
        self._check('archive_results', self._archive_dir, 'comparison_experiment_dir')
        archive_name = archive_name or f'comparison_results_{self.timestamp}'
        archive_path = os.path.join(self._archive_dir, f'{archive_name}.zip')

        collected = 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for dirpath, _, filenames in os.walk(self._comparison_experiment_dir):
                for filename in filenames:
                    if filename.endswith('.zip'):
                        continue
                    filepath = os.path.join(dirpath, filename)
                    arcname  = os.path.relpath(
                        filepath, os.path.dirname(self._comparison_experiment_dir))
                    zout.write(filepath, arcname)
                    collected += 1

        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f'  Archived {collected} files -> {archive_path} ({size_mb:.2f} MB)')
        return archive_path

    # ── download to local ─────────────────────────────────────────────────────

    def download_archive(self, archive_path: Optional[str] = None) -> None:
        """Downloads the most recent (or specified) archive to local via Colab."""
        self._check('download_archive', self._archive_dir, 'comparison_experiment_dir')
        if archive_path is None:
            zips = sorted(
                [f for f in os.listdir(self._archive_dir) if f.endswith('.zip')],
                key=lambda f: os.path.getmtime(os.path.join(self._archive_dir, f))
            )
            if not zips:
                raise FileNotFoundError(
                    f'No archives found in {self._archive_dir}. '
                    'Call archive_results() first.'
                )
            archive_path = os.path.join(self._archive_dir, zips[-1])

        if not os.path.exists(archive_path):
            raise FileNotFoundError(f'Archive not found: {archive_path}')

        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f'Downloading {os.path.basename(archive_path)} ({size_mb:.2f} MB) ...')
        files.download(archive_path)

    # ── guard ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _check(method: str, value, attr: str) -> None:
        if value is None:
            raise RuntimeError(
                f'{method}() requires {attr} to be set first.'
            )