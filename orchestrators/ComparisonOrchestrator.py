import zipfile
from pathlib import Path
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
        self.train_experiments_dir    = Path(train_experiments_dir)
        self._archive_dir             = Path(archive_dir)
        self._keras_visualizations_dir  = Path(keras_visualizations_dir)
        self._tflite_visualizations_dir = Path(tflite_visualizations_dir)

        self.train_experiments_dir.mkdir(parents=True, exist_ok=True)
        self._zips_dir     = self.train_experiments_dir / 'zips'
        self._unpacked_dir = self.train_experiments_dir / 'unpacked'
        self._zips_dir.mkdir(exist_ok=True)
        self._unpacked_dir.mkdir(exist_ok=True)

        print('ComparisonOrchestrator initialized.')
        print(f'  zips      -> {self._zips_dir}')
        print(f'  unpacked  -> {self._unpacked_dir}')
        print(f'  archive   -> {self._archive_dir}')

    @property
    def unpacked_dir(self) -> Path:
        """Path to unpacked experiment folders — pass to handler constructors."""
        return self._unpacked_dir

    # ── download ──────────────────────────────────────────────────────────────

    def download_experiments(self, source: str) -> list:
        """Downloads experiment ZIPs from a Google Drive folder share link."""
        print(f'Downloading experiments from Google Drive -> {self._zips_dir}')
        gdown.download_folder(
            url=source,
            output=str(self._zips_dir),
            quiet=False,
            use_cookies=False,
        )
        zips = sorted(self._zips_dir.glob('*.zip'))
        print(f'  Download complete — {len(zips)} ZIPs in {self._zips_dir}')
        return zips

    # ── unzip ─────────────────────────────────────────────────────────────────

    def unzip_experiments(self) -> list:
        """Unpacks ZIPs from zips/ into unpacked/, one folder per experiment. Skips existing."""
        zips = sorted(self._zips_dir.glob('*.zip'))
        if not zips:
            raise FileNotFoundError(
                f'No ZIPs found in {self._zips_dir}. '
                'Call download_experiments() first.'
            )
        print(f'Unpacking {len(zips)} experiment ZIPs ...')
        unpacked = []
        for zip_path in zips:
            dest = self._unpacked_dir / zip_path.stem
            if dest.exists():
                print(f'  [skip] {zip_path.name} (already unpacked)')
                unpacked.append(dest)
                continue
            dest.mkdir()
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest)
            print(f'  {zip_path.name} -> {dest.name}/')
            unpacked.append(dest)
        print(f'  {len(unpacked)} experiments ready in {self._unpacked_dir}')
        return unpacked

    # ── archive ───────────────────────────────────────────────────────────────

    def archive_results(self, archive_name: Optional[str] = None) -> Path:
        """Zips all plots from both visualization dirs into archive_dir."""
        timestamp    = _dt.now().strftime('%Y%m%d-%H%M%S')
        archive_name = archive_name or f'comparison_plots_{timestamp}'
        archive_path = self._archive_dir / f'{archive_name}.zip'

        collected = 0
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for plot_dir in (self._keras_visualizations_dir, self._tflite_visualizations_dir):
                for png in sorted(plot_dir.glob('*.png')):
                    zout.write(png, arcname=f'{plot_dir.name}/{png.name}')
                    collected += 1

        size_kb = archive_path.stat().st_size / 1024
        print(f'  Archived {collected} plots -> {archive_path} ({size_kb:.1f} KB)')
        return archive_path

    # ── download to local ─────────────────────────────────────────────────────

    def download_archive(self, archive_path: Optional[Path] = None) -> None:
        """Downloads the most recent (or specified) archive to local via Colab."""
        if archive_path is None:
            zips = sorted(self._archive_dir.glob('*.zip'),
                          key=lambda p: p.stat().st_mtime)
            if not zips:
                raise FileNotFoundError(
                    f'No archives found in {self._archive_dir}. '
                    'Call archive_results() first.'
                )
            archive_path = zips[-1]

        if not archive_path.exists():
            raise FileNotFoundError(f'Archive not found: {archive_path}')

        size_kb = archive_path.stat().st_size / 1024
        print(f'Downloading {archive_path.name} ({size_kb:.1f} KB) ...')
        files.download(str(archive_path))


print('ComparisonOrchestrator defined.')