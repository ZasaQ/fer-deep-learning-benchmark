import os
from typing import Optional


class DirectoryManager:
    """Manages experiment directory structure."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.root: Optional[str] = None
        self.paths: dict         = {}

        print('DirectoryManager has been initialized.')

    # ── public ────────────────────────────────────────────────────────────────

    def create_experiment_dirs(
        self,
        visualization_subdirs: list[str],
        other_subdirs: list[str],
    ) -> str:
        """Create the full experiment directory structure."""
        self.root = self.experiment_name
        os.makedirs(self.root, exist_ok=True)
        self.paths = {'root': self.root}

        visualizations_root = os.path.join(self.root, 'visualizations')
        os.makedirs(visualizations_root, exist_ok=True)
        self.paths['visualizations'] = visualizations_root

        for subdir in visualization_subdirs:
            path = os.path.join(visualizations_root, subdir)
            os.makedirs(path, exist_ok=True)
            self.paths[subdir] = path

        for subdir in other_subdirs:
            path = os.path.join(self.root, subdir)
            os.makedirs(path, exist_ok=True)
            self.paths[subdir] = path

        self._print_structure(visualization_subdirs, other_subdirs)
        return self.root

    def create_dir(self, path: str, key: Optional[str] = None) -> str:
        """Create an arbitrary directory and register it in paths."""
        os.makedirs(path, exist_ok=True)
        registered_key = key or os.path.basename(path)
        self.paths[registered_key] = path
        print(f'  {registered_key} -> {path}')
        return path

    def get(self, subdir: str) -> str:
        if not self.paths:
            raise RuntimeError('Directories not created yet. Call create_experiment_dirs() first.')
        if subdir not in self.paths:
            raise KeyError(
                f"Unknown subdir: '{subdir}'. "
                f"Available: {list(self.paths.keys())}"
            )
        return self.paths[subdir]

    def list_contents(self, subdir: str = 'root') -> None:
        """List files in a specific subdirectory with sizes."""
        path = self.get(subdir)
        if not os.path.exists(path):
            print(f'Directory not found: {path}')
            return

        entries = sorted(os.listdir(path))
        if not entries:
            print(f'  {subdir}/ is empty')
            return

        print(f'\n  {subdir}/')
        for f in entries:
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                size_kb = os.path.getsize(fp) / 1024
                print(f'    {f:<40} {size_kb:>8.1f} KB')
            else:
                print(f'    {f}/')

    def list_all_contents(self) -> None:
        """List files across all subdirectories."""
        if not self.paths:
            print('Directories not created yet.')
            return
        print(f'\nExperiment: {self.root}')
        print('=' * 55)
        for subdir in self.paths:
            self.list_contents(subdir)

    def total_size_mb(self) -> float:
        """Return total size of all files in the experiment directory."""
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 * 1024)

    # ── private ───────────────────────────────────────────────────────────────

    def _print_structure(
        self,
        visualization_subdirs: list[str],
        other_subdirs: list[str],
    ) -> None:
        print(f'\n{self.root}/')
        print(f'   ├── visualizations/')
        for i, subdir in enumerate(visualization_subdirs):
            connector = '└' if i == len(visualization_subdirs) - 1 else '├'
            print(f'   │   {connector}── {subdir}/')
        for i, subdir in enumerate(other_subdirs):
            connector = '└' if i == len(other_subdirs) - 1 else '├'
            print(f'   {connector}── {subdir}/')
        print()