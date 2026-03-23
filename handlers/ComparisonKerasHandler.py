import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .BaseComparisonHandler import BaseComparisonHandler


class ComparisonKerasHandler(BaseComparisonHandler):
    """Handler for visualizing and comparing Keras training results across all experiments."""

    VISUALIZATION_SUBDIRS = ['comparison_keras_visualizations']

    METRICS_HEATMAP = [
        ('test_accuracy',   'Test Accuracy'),
        ('test_f1_macro',   'Macro F1 Score'),
        ('precision_macro', 'Macro Precision'),
        ('recall_macro',    'Macro Recall'),
        ('macro_auc',       'Macro AUC'),
    ]

    def __init__(
        self,
        train_experiments_dir: str,
        visualizations_directory: str,
    ):
        super().__init__(
            train_experiments_dir=train_experiments_dir,
            visualizations_directory=visualizations_directory,
        )
        print('ComparisonKerasHandler initialized.')

    # ── public ────────────────────────────────────────────────────────────────

    def plot_metrics_heatmap(self, figsize=(14, 8)) -> None:
        """Generate one heatmap per metric from METRICS_HEATMAP."""
        self._check_loaded()
        for metric, label in self.METRICS_HEATMAP:
            if metric not in self.df.columns:
                print(f'Column "{metric}" not in DataFrame')
                continue
            self._build_metric_heatmap(
                metric,
                f'{label}  —  Model × Strategy × Dataset',
                f'{metric}_heatmap.png',
                figsize,
            )

    def plot_learning_curves_overlay(self, figsize=(16, 10)) -> None:
        self._check_loaded()

        datasets = [d for d in self.DATASET_ORDER
                    if any(r.dataset == d for r in self.records)]
        n_cols = 2
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle('Validation Accuracy — Learning Curves Overlay',
                     fontsize=14, fontweight='bold', y=1.01)
        for idx, dataset in enumerate(datasets):
            ax = axes[idx // n_cols][idx % n_cols]
            for rec in self.records:
                if rec.dataset != dataset:
                    continue
                curve = rec.history.get('val_accuracy') or rec.history.get('val_acc')
                if not curve:
                    continue
                ax.plot(range(1, len(curve) + 1), curve,
                        color=self.MODEL_COLORS.get(rec.model, '#888'),
                        linestyle=self.STRATEGY_LS.get(rec.strategy, '-'),
                        linewidth=1.4, alpha=0.82)
            ax.set_title(dataset, fontsize=11)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val Accuracy')
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        for idx in range(len(datasets), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
        handles = []
        for model, color in self.MODEL_COLORS.items():
            handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=model))
        for strat, ls in self.STRATEGY_LS.items():
            handles.append(plt.Line2D([0], [0], color='black', linestyle=ls,
                                      linewidth=1.5, label=strat))
        fig.legend(handles=handles, loc='lower center',
                   ncol=len(self.MODEL_COLORS) + len(self.STRATEGY_LS),
                   bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
        self._save_fig('learning_curves_overlay.png')

    def plot_per_class_f1_comparison(self, figsize=(15, 9)) -> None:
        self._check_loaded()

        datasets = [d for d in self.DATASET_ORDER
                    if any(r.dataset == d for r in self.records)]
        n_cols = 2
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle('Per-Class F1 Score  —  Model Comparison by Dataset',
                     fontsize=13, fontweight='bold')
        for idx, dataset in enumerate(datasets):
            ax     = axes[idx // n_cols][idx % n_cols]
            subset = self.df[self.df['dataset'] == dataset]
            emotions_avail = [e for e in self.EMOTION_CLASSES
                              if f'f1_{e.lower()}' in subset.columns
                              and subset[f'f1_{e.lower()}'].notna().any()]
            if not emotions_avail:
                ax.text(0.5, 0.5, 'No per-class F1 data',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset, fontsize=11)
                continue
            models_avail = [m for m in self.MODEL_ORDER if m in subset['model'].values]
            x     = np.arange(len(emotions_avail))
            width = 0.8 / max(len(models_avail), 1)
            for i, model in enumerate(models_avail):
                mrows  = subset[subset['model'] == model]
                means  = [mrows[f'f1_{e.lower()}'].mean() for e in emotions_avail]
                offset = (i - len(models_avail) / 2 + 0.5) * width
                ax.bar(x + offset, means, width,
                       label=model, color=self.MODEL_COLORS.get(model, '#888'),
                       edgecolor='white', linewidth=0.5, alpha=0.9)
            ax.set_title(dataset, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(emotions_avail, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('F1 Score')
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        for idx in range(len(datasets), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=self.MODEL_COLORS.get(m, '#888'), label=m)
                   for m in self.MODEL_ORDER]
        fig.legend(handles=handles, loc='lower center', ncol=len(self.MODEL_ORDER),
                   bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)
        self._save_fig('per_class_f1_comparison.png')

    # ── plot builders ─────────────────────────────────────────────────────────

    def _build_metric_heatmap(
        self, metric: str, title: str, filename: str, figsize: tuple,
    ) -> None:
        df = self.df[['model', 'dataset', 'strategy', metric]].dropna(subset=[metric])
        if df.empty:
            print(f'  [skip] No data for metric "{metric}"')
            return
        pivot = df.pivot_table(
            index=['model', 'strategy'], columns='dataset',
            values=metric, aggfunc='mean',
        )
        idx_order = [(m, s) for m in self.MODEL_ORDER for s in self.STRATEGY_ORDER
                     if (m, s) in pivot.index]
        col_order = [d for d in self.DATASET_ORDER if d in pivot.columns]
        pivot = pivot.reindex(idx_order)[col_order]

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=13, fontweight='bold')
        mask = pivot.isna()
        sns.heatmap(pivot, ax=ax, mask=mask,
                    annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1,
                    linewidths=0.5, linecolor='#dddddd',
                    cbar_kws={'label': metric.replace('_', ' ').title(), 'shrink': 0.75})
        if mask.any().any():
            sns.heatmap(pivot, ax=ax, mask=~mask,
                        annot=False, cmap=['#eeeeee'], vmin=0, vmax=1,
                        linewidths=0.5, linecolor='#dddddd', cbar=False)
        cumsum = 0
        for m in self.MODEL_ORDER:
            cnt = sum(1 for (mm, _) in idx_order if mm == m)
            cumsum += cnt
            if cumsum < len(idx_order):
                ax.axhline(cumsum, color='black', linewidth=1.8)
        ax.set_xlabel('Dataset', labelpad=8)
        ax.set_ylabel('Model / Strategy', labelpad=8)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
        self._save_fig(filename)