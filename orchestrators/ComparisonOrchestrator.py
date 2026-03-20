import os
import zipfile
from datetime import datetime as _dt
from typing import Optional

import gdown
from google.colab import files


class ComparisonOrchestrator:
    """Handles Google Drive download, ZIP unpacking, plot archiving and local download."""

    def __init__(
        self,
        train_experiments_dir: str,
        comparison_experiment_dir: str,
    ):
        self.train_experiments_dir     = train_experiments_dir
        self.comparison_experiment_dir = comparison_experiment_dir
        self._archive_dir              = os.path.join(comparison_experiment_dir, 'archive')

        self._zips_dir     = os.path.join(train_experiments_dir, 'zips')
        self._unpacked_dir = os.path.join(train_experiments_dir, 'unpacked')

        os.makedirs(train_experiments_dir, exist_ok=True)
        os.makedirs(self._zips_dir,        exist_ok=True)
        os.makedirs(self._unpacked_dir,    exist_ok=True)

        print('ComparisonOrchestrator initialized.')
        print(f'  zips      -> {self._zips_dir}')
        print(f'  unpacked  -> {self._unpacked_dir}')
        print(f'  archive   -> {self._archive_dir}')

    @property
    def unpacked_dir(self) -> str:
        """Path to unpacked experiment folders — pass to handler constructors."""
        return self._unpacked_dir

    # ── download ──────────────────────────────────────────────────────────────

    def download_experiments(self, source: str) -> list:
        """Downloads experiment ZIPs from a Google Drive folder share link."""
        print(f'Downloading experiments from Google Drive -> {self._zips_dir}')
        gdown.download_folder(
            url=source,
            output=self._zips_dir,
            quiet=False,
            use_cookies=False,
        )
        zips = sorted([f for f in os.listdir(self._zips_dir) if f.endswith('.zip')])
        print(f'  Download complete — {len(zips)} ZIPs in {self._zips_dir}')
        return zips

    # ── unzip ─────────────────────────────────────────────────────────────────

    def unzip_experiments(self) -> list:
        """Unpacks ZIPs from zips/ into unpacked/, one folder per experiment. Skips existing."""
        zips = sorted([f for f in os.listdir(self._zips_dir) if f.endswith('.zip')])
        if not zips:
            raise FileNotFoundError(
                f'No ZIPs found in {self._zips_dir}. '
                'Call download_experiments() first.'
            )
        print(f'Unpacking {len(zips)} experiment ZIPs ...')
        unpacked = []
        for zip_name in zips:
            zip_path = os.path.join(self._zips_dir, zip_name)
            dest     = os.path.join(self._unpacked_dir, os.path.splitext(zip_name)[0])
            if os.path.exists(dest):
                print(f'  [skip] {zip_name} (already unpacked)')
                unpacked.append(dest)
                continue
            os.makedirs(dest)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            print(f'  {zip_name} -> {os.path.basename(dest)}/')
            unpacked.append(dest)
        print(f'  {len(unpacked)} experiments ready in {self._unpacked_dir}')
        return unpacked

    # ── archive ───────────────────────────────────────────────────────────────

    def archive_results(self, archive_name: Optional[str] = None) -> str:
        """Zips the entire comparison experiment folder into archive/."""
        timestamp    = _dt.now().strftime('%Y%m%d-%H%M%S')
        archive_name = archive_name or f'comparison_results_{timestamp}'
        archive_path = os.path.join(self._archive_dir, f'{archive_name}.zip')

        collected = 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for dirpath, _, filenames in os.walk(self.comparison_experiment_dir):
                for filename in filenames:
                    if filename.endswith('.zip'):
                        continue
                    filepath = os.path.join(dirpath, filename)
                    arcname  = os.path.relpath(filepath, os.path.dirname(self.comparison_experiment_dir))
                    zout.write(filepath, arcname)
                    collected += 1

        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f'  Archived {collected} files -> {archive_path} ({size_mb:.2f} MB)')
        return archive_path

    # ── download to local ─────────────────────────────────────────────────────

    def download_archive(self, archive_path: Optional[str] = None) -> None:
        """Downloads the most recent (or specified) archive to local via Colab."""
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
