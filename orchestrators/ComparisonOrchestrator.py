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
        archive_dir: str,
        keras_visualizations_dir: str,
        tflite_visualizations_dir: str,
    ):
        self.train_experiments_dir     = train_experiments_dir
        self._archive_dir              = archive_dir
        self._keras_visualizations_dir  = keras_visualizations_dir
        self._tflite_visualizations_dir = tflite_visualizations_dir

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
        zips = sorted([
            f for f in os.listdir(self._zips_dir) if f.endswith('.zip')
        ])
        print(f'  Download complete — {len(zips)} ZIPs in {self._zips_dir}')
        return zips

    # ── unzip ─────────────────────────────────────────────────────────────────

    def unzip_experiments(self) -> list:
        """Unpacks ZIPs from zips/ into unpacked/, one folder per experiment. Skips existing."""
        zips = sorted([
            f for f in os.listdir(self._zips_dir) if f.endswith('.zip')
        ])
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
        """Zips all plots from both visualization dirs into archive_dir."""
        timestamp    = _dt.now().strftime('%Y%m%d-%H%M%S')
        archive_name = archive_name or f'comparison_plots_{timestamp}'
        archive_path = os.path.join(self._archive_dir, f'{archive_name}.zip')

        collected = 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for plot_dir in (self._keras_visualizations_dir, self._tflite_visualizations_dir):
                for png in sorted(os.listdir(plot_dir)):
                    if not png.endswith('.png'):
                        continue
                    full_path = os.path.join(plot_dir, png)
                    arcname   = os.path.join(os.path.basename(plot_dir), png)
                    zout.write(full_path, arcname)
                    collected += 1

        size_kb = os.path.getsize(archive_path) / 1024
        print(f'  Archived {collected} plots -> {archive_path} ({size_kb:.1f} KB)')
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

        size_kb = os.path.getsize(archive_path) / 1024
        print(f'Downloading {os.path.basename(archive_path)} ({size_kb:.1f} KB) ...')
        files.download(archive_path)


print('ComparisonOrchestrator defined.')