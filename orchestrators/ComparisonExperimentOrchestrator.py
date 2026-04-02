import os
import shutil
import zipfile
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

    # ── validation ────────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        """Check if all required directories and handlers are configured."""
        checks = {
            'comparison_experiment_dir': self.comparison_experiment_dir is not None,
            'trained_zips_dir':          self.trained_zips_dir is not None,
            'trained_unpacked_dir':      self.trained_unpacked_dir is not None,
            'keras_handler':             self._keras_handler is not None,
            'tflite_handler':            self._tflite_handler is not None,
        }
        all_ok = all(checks.values())
        print('\ComparisonExperimentOrchestrator status:')
        for name, ok in checks.items():
            print(f'  {"Ok" if ok else "Error"} {name}')
        return all_ok

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
            raise FileNotFoundError(
                f'Drive source directory not found: {source_dir}\n'
                'Call mount_drive() first.'
            )

        os.makedirs(self.trained_zips_dir, exist_ok=True)

        available = sorted([f for f in os.listdir(source_dir) if f.endswith('.zip')])
        if pattern:
            available = [f for f in available if pattern in f]

        if not available:
            raise FileNotFoundError(
                f'No ZIPs found in {source_dir}'
                + (f' matching pattern "{pattern}"' if pattern else '')
            )

        print(f'Copying {len(available)} ZIPs from Drive -> {self.trained_zips_dir}')
        copied, skipped = [], []
        for filename in available:
            src  = os.path.join(source_dir, filename)
            dest = os.path.join(self.trained_zips_dir, filename)
            if os.path.exists(dest):
                print(f'{filename} (already exists)')
                skipped.append(dest)
                continue
            shutil.copy2(src, dest)
            print(f'  {filename}')
            copied.append(dest)

        print(f'  Copied {len(copied)}, skipped {len(skipped)} — '
              f'{len(copied) + len(skipped)} ZIPs total in {self.trained_zips_dir}')
        return copied + skipped

    # ── unzip ─────────────────────────────────────────────────────────────────

    def unzip_experiments(self) -> list:
        self._check('unzip_experiments', self.trained_zips_dir,     'zips_dir')
        self._check('unzip_experiments', self.trained_unpacked_dir, 'unpacked_dir')

        zips = sorted([f for f in os.listdir(self.trained_zips_dir) if f.endswith('.zip')])
        if not zips:
            raise FileNotFoundError(
                f'No ZIPs found in {self.trained_zips_dir}. '
                'Call copy_experiments() first.'
            )

        print(f'Unpacking {len(zips)} experiment ZIPs ...')
        unpacked = []
        for zip_name in zips:
            zip_path = os.path.join(self.trained_zips_dir, zip_name)
            dest     = os.path.join(self.trained_unpacked_dir, os.path.splitext(zip_name)[0])
            if os.path.exists(dest):
                print(f'{zip_name} (already unpacked)')
                unpacked.append(dest)
                continue
            os.makedirs(dest)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            print(f'  {zip_name} -> {os.path.basename(dest)}/')
            unpacked.append(dest)

        print(f'  {len(unpacked)} experiments ready in {self.trained_unpacked_dir}')
        return unpacked

    # ── archive ───────────────────────────────────────────────────────────────

    def archive_results(self, archive_name: Optional[str] = None) -> str:
        self._check('archive_results', self._archive_dir, 'comparison_experiment_dir')

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