import os
from typing import Optional, List


class BaseHandler:
    """
    Shared base for all FER experiment handlers.
    Provides guard checks, summary table printing and archive directory management.
    Handlers that produce plots should inherit from BaseVisualizer instead.
    """

    def __init__(self, visualizations_directory: Optional[str] = None):
        self.visualizations_directory = visualizations_directory
        self._archive_directory: Optional[str] = None
        self._experiment_orchestrator = None

    # ── archive directory ────────────────────────────────────

    @property
    def archive_directory(self) -> Optional[str]:
        """Lazy lookup: own value first, then via ExperimentOrchestrator back-reference."""
        if self._archive_directory is not None:
            return self._archive_directory
        if self._experiment_orchestrator is not None:
            return self._experiment_orchestrator.archive_directory
        return None

    @archive_directory.setter
    def archive_directory(self, value: str) -> None:
        self._archive_directory = value

    # ── guard ────────────────────────────────────────────────

    def _guard(self, condition: bool, message: str) -> bool:
        """Print message and return False if condition is not met."""
        if not condition:
            print(f'{message}')
        return condition

    # ── summary table ────────────────────────────────────────

    def _generate_ascii_summary(self, title: str, sections: list) -> None:
        """
        Print a two-column summary table with optional section separators.
        """
        rows = [(k, v) for item in sections if item is not None for k, v in [item] if v is not None]
        col_key = max((len(k) for k, v in rows), default=20) + 2
        col_val = max((len(str(v)) for k, v in rows), default=30) + 2
        col_key = max(col_key, len(title) + 2, 26)
        col_val = max(col_val, 32)
        sep = '+' + '-' * col_key + '+' + '-' * col_val + '+'

        def fmt_row(k, v):
            return f'| {k:<{col_key-2}} | {v:<{col_val-2}} |'

        print()
        print(sep)
        print(f'| {title:<{col_key-2}} | {"":>{col_val-2}} |')
        print(sep)
        for item in sections:
            if item is None:
                print(sep)
            else:
                k, v = item
                if v is not None:
                    print(fmt_row(str(k), str(v)))
        print(sep)
        print()

    def _generate_latex_summary(self, title: str, sections: list, filename: str) -> None:
        """
        Print and optionally save LaTeX summary table to archive_directory.
        """
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{ll}")
        lines.append("\\toprule")
        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        lines.append("\\midrule")

        for item in sections:
            if item is None:
                lines.append("\\midrule")
            else:
                k, v = item
                if v is not None:
                    clean_k = str(k).replace("_", "\\_")
                    clean_v = str(v).replace("%", "\\%").replace("_", "\\_")
                    lines.append(f"{clean_k} & {clean_v} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{Summary table for {title}}}")
        lines.append("\\end{table}")

        output = "\n".join(lines)
        print(output)

        if self.archive_directory:
            path = os.path.join(self.archive_directory, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(output + "\n")
            print(f"LaTeX saved to: {path}")
        else:
            print(f"archive_directory not set — LaTeX not saved.")

    def print_summary(self, mode: str = 'ascii') -> None:
        """Print handler state summary. Override in subclasses."""
        raise NotImplementedError(f'{self.__class__.__name__}.print_summary() is not implemented.')