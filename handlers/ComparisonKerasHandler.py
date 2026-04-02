import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .BaseComparisonHandler import BaseComparisonHandler


class ComparisonKerasHandler(BaseComparisonHandler):
    """Handler for loading, processing and visualizing Keras training experiment results in the comparison experiment context."""

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

    # ── visualization ────────────────────────────────────────────────────────

    def plot_metrics_heatmap(self, figsize=(14, 8)) -> None:
        """Plot a heatmap comparing test accuracy across models and strategies for each dataset."""
        self._check_loaded()

        metrics = [
            (m, l) for m, l in self.METRICS_HEATMAP
            if m in self.df.columns and self.df[m].notna().any()
        ]
        if not metrics:
            print('No metric columns available.')
            return

        for metric, label in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(label, y=1.01)

            df = self.df[['model', 'dataset', 'strategy', metric]].dropna(subset=[metric])
            pivot = df.pivot_table(
                index=['model', 'strategy'], columns='dataset',
                values=metric, aggfunc='mean',
                observed=False,
            )
            idx_order = [(m, s) for m in self.MODEL_ORDER for s in self.STRATEGY_ORDER
                         if (m, s) in pivot.index]
            col_order = [d for d in self.DATASET_ORDER if d in pivot.columns]
            pivot = pivot.reindex(idx_order)[col_order]

            mask = pivot.isna()
            sns.heatmap(pivot, ax=ax, mask=mask,
                        annot=True, fmt='.3f', cmap='Blues', vmin=0, vmax=1,
                        linewidths=0.5, linecolor='#dddddd',
                        cbar_kws={'label': label, 'shrink': 0.75})
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

            ax.set_xlabel('Dataset', labelpad=6)
            ax.set_ylabel('Model-Strategy', labelpad=6)
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', rotation=0)

            plt.tight_layout()
            self._save_fig(f'metrics_heatmap_{metric}.png')
            plt.close(fig)

    def plot_accuracy_bar_chart(self, figsize=(15, 9)) -> None:
        """Plot grouped bar charts comparing test accuracy across models and strategies for each dataset."""
        self._check_loaded()

        metrics = [m for m, _ in self.METRICS_HEATMAP
                   if m in self.df.columns and self.df[m].notna().any()]
        if not metrics:
            print('No metric columns available.')
            return

        labels       = {m: l for m, l in self.METRICS_HEATMAP}
        metric       = metrics[0]
        label        = labels.get(metric, metric)
        datasets     = [d for d in self.DATASET_ORDER if d in self.df['dataset'].values]
        strategies   = [s for s in self.STRATEGY_ORDER if s in self.df['strategy'].values]
        models_avail = [m for m in self.MODEL_ORDER if m in self.df['model'].values]

        n_cols = 2
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle(f'{label} | Model × Strategy by Dataset', y=1.01)

        x     = np.arange(len(models_avail))
        width = 0.8 / max(len(strategies), 1)

        for idx, dataset in enumerate(datasets):
            ax     = axes[idx // n_cols][idx % n_cols]
            subset = self.df[self.df['dataset'] == dataset]

            for i, strategy in enumerate(strategies):
                srows  = subset[subset['strategy'] == strategy]
                values = [
                    srows[srows['model'] == m][metric].mean()
                    if m in srows['model'].values else np.nan
                    for m in models_avail
                ]
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, values, width,
                            label=strategy,
                            color=self.STRATEGY_COLORS.get(strategy, '#888'),
                            edgecolor='white', linewidth=0.5, alpha=0.88)
                ax.bar_label(bars,
                            labels=[f'{v:.0%}' if not np.isnan(v) else '' for v in values],
                            fontsize=8, padding=2, rotation=0, fontweight='bold',
                            color='#444444')


            ax.set_title(dataset, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(models_avail, rotation=20, ha='right', fontsize=9)
            ax.set_ylabel(label if idx % n_cols == 0 else '', fontsize=9)
            ax.set_ylim(0, 1.10)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(datasets), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=self.STRATEGY_COLORS.get(s, '#888'), label=s)
                   for s in strategies]
        fig.legend(handles=handles, loc='lower center', ncol=len(strategies),
                   bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=10)

        plt.tight_layout()
        self._save_fig('accuracy_bar_chart.png')

    def plot_learning_curves_overlay(self, figsize=(20, 14)) -> None:
        """Plot overlaid learning curves of validation accuracy across epochs for all model-strategy combinations, faceted by dataset."""
        self._check_loaded()

        datasets   = [d for d in self.DATASET_ORDER   if any(r.dataset   == d for r in self.records)]
        strategies = [s for s in self.STRATEGY_ORDER  if any(r.strategy  == s for r in self.records)]
        n_rows = len(datasets)
        n_cols = len(strategies)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                squeeze=False, sharex=False, sharey=True)
        fig.suptitle('Validation Accuracy Overlay', y=1.01)

        for r_idx, dataset in enumerate(datasets):
            for c_idx, strategy in enumerate(strategies):
                ax = axes[r_idx][c_idx]
                for rec in self.records:
                    if rec.dataset != dataset or rec.strategy != strategy:
                        continue
                    curve = rec.history.get('val_accuracy') or rec.history.get('val_acc')
                    if not curve:
                        continue
                    ax.plot(range(1, len(curve) + 1), curve,
                            color=self.MODEL_COLORS.get(rec.model, '#888'),
                            linewidth=1.4, alpha=0.82)

                ax.margins(x=0, y=0)
                ax.grid(axis='y', alpha=0.3)

                if r_idx == 0:
                    ax.set_title(strategy, fontsize=10)
                if c_idx == 0:
                    ax.set_ylabel(f'{dataset}\nVal Accuracy', fontsize=9)
                ax.set_xlabel('Epoch', fontsize=8)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
                ax.tick_params(labelsize=8)

        handles = [plt.Line2D([0], [0], color=self.MODEL_COLORS.get(m, '#888'),
                            linewidth=2, label=m)
                for m in self.MODEL_ORDER]
        fig.legend(handles=handles, loc='lower center', ncol=len(self.MODEL_ORDER),
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
        plt.tight_layout()
        for row_axes in axes:
            for ax in row_axes:
                ax.tick_params(labelleft=True)        

        self._save_fig('learning_curves_overlay.png')

    def plot_per_class_f1_comparison(self, figsize=(15, 9)) -> None:
        """Plot grouped bar charts comparing per-class F1 scores across models for each dataset."""
        self._check_loaded()

        datasets = [d for d in self.DATASET_ORDER
                    if any(r.dataset == d for r in self.records)]
        n_cols = 2
        n_rows = int(np.ceil(len(datasets) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle('Per-Class F1 Score by Dataset')

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
                bars = ax.bar(x + offset, means, width,
                       label=model, color=self.MODEL_COLORS.get(model, '#888'),
                       edgecolor='white', linewidth=0.5, alpha=0.88)
                ax.bar_label(bars,
                            labels=[f'{v:.0%}' if not np.isnan(v) else '' for v in means],
                            fontsize=8, padding=2, rotation=0, fontweight='bold',
                            color='#444444')
            ax.set_title(dataset, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(emotions_avail, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('F1 Score' if idx % n_cols == 0 else '')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        for idx in range(len(datasets), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=self.MODEL_COLORS.get(m, '#888'), label=m)
                   for m in self.MODEL_ORDER]
        fig.legend(handles=handles, loc='lower center', ncol=len(self.MODEL_ORDER),
                   bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=10)

        plt.tight_layout()
        self._save_fig('per_class_f1_comparison.png')

    def plot_train_val_gap_heatmap(self, figsize=(14, 8)) -> None:
        """Plot a heatmap of the train-validation accuracy gap across models and strategies, faceted by dataset."""
        self._check_loaded()

        if 'train_val_gap' not in self.df.columns or self.df['train_val_gap'].isna().all():
            print('No train_val_gap data available.')
            return

        df = self.df[['model', 'dataset', 'strategy', 'train_val_gap',
                      'test_accuracy', 'actual_epochs']].copy()
        df = df.dropna(subset=['train_val_gap'])

        pivot = df.pivot_table(
            index=['model', 'strategy'], columns='dataset',
            values='train_val_gap', aggfunc='mean',
            observed=False,
        )
        idx_order = [(m, s) for m in self.MODEL_ORDER for s in self.STRATEGY_ORDER
                     if (m, s) in pivot.index]
        col_order = [d for d in self.DATASET_ORDER if d in pivot.columns]
        pivot = pivot.reindex(idx_order)[col_order]

        abs_max = max(pivot.abs().max().max(), 0.01)

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Train–Val Accuracy Gap', y=1.01)

        mask = pivot.isna()
        sns.heatmap(pivot, ax=ax, mask=mask,
                    annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-abs_max, vmax=abs_max,
                    linewidths=0.5, linecolor='#dddddd',
                    cbar_kws={'label': 'Train Accuracy − Val Accuracy', 'shrink': 0.75})
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

        ax.set_xlabel('Dataset', labelpad=6)
        ax.set_ylabel('Model-Strategy', labelpad=6)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        self._save_fig('train_val_gap_heatmap.png')

    def plot_epochs_stripplot(self, figsize=(14, 8)) -> None:
        """Plot a strip plot of actual epochs trained across models and strategies, faceted by dataset."""
        self._check_loaded()

        needed = ['actual_epochs', 'best_val_accuracy', 'model', 'strategy', 'dataset']
        missing = [c for c in needed if c not in self.df.columns]
        if missing or self.df['actual_epochs'].isna().all():
            print(f'Missing columns for convergence plot: {missing}')
            return

        df = self.df[needed].dropna(subset=['actual_epochs', 'best_val_accuracy'])

        models     = [m for m in self.MODEL_ORDER    if m in df['model'].values]
        strategies = [s for s in self.STRATEGY_ORDER if s in df['strategy'].values]

        n_cols = 2
        n_rows = int(np.ceil(len(strategies) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False,
                                sharey=True)
        fig.suptitle('Epochs Trained | Model × Strategy')

        x     = np.arange(len(models))
        rng   = np.random.default_rng(42)

        for idx, strategy in enumerate(strategies):
            ax     = axes[idx // n_cols][idx % n_cols]
            subset = df[df['strategy'] == strategy]

            for i, model in enumerate(models):
                vals = subset[subset['model'] == model]['actual_epochs'].dropna().values
                if len(vals) == 0:
                    continue
                color  = self.MODEL_COLORS.get(model, '#888')
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(i + jitter, vals, color=color,
                        s=55, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)
                ax.hlines(np.median(vals), i - 0.25, i + 0.25,
                        colors=color, linewidths=2.0, zorder=4)

            ax.set_title(strategy, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
            ax.set_ylabel('Epochs Trained' if idx % n_cols == 0 else '', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        for idx in range(len(strategies), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        model_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=self.MODEL_COLORS.get(m, '#888'),
                    markersize=8, label=m)
            for m in models
        ]
        fig.legend(handles=model_handles, loc='lower center',
                ncol=len(models), bbox_to_anchor=(0.5, -0.02),
                frameon=False, fontsize=9)

        plt.tight_layout()
        for row_axes in axes:
            for ax in row_axes:
                ax.tick_params(labelleft=True)
        plt.subplots_adjust(bottom=0.10)
        self._save_fig('epochs_stripplot.png')

    def plot_calibration_comparison(self, figsize=(16, 8)) -> None:
        """Plot side-by-side bar charts comparing ECE and Brier Score across models and strategies for each dataset."""
        self._check_loaded()

        metrics_avail = [
            (col, label) for col, label in [('ece', 'ECE'), ('brier_score', 'Brier Score')]
            if col in self.df.columns and self.df[col].notna().any()
        ]
        if not metrics_avail:
            print('No calibration metrics (ece / brier_score) available.')
            return

        datasets     = [d for d in self.DATASET_ORDER if d in self.df['dataset'].values]
        models_avail = [m for m in self.MODEL_ORDER   if m in self.df['model'].values]

        n_metrics = len(metrics_avail)
        n_cols    = 2
        n_rows    = int(np.ceil(len(datasets) * n_metrics / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        ylims = {}
        for col, label in metrics_avail:
            max_val = self.df[col].max(skipna=True)
            ylims[col] = max_val * 1.25 if pd.notna(max_val) else 0.1
        fig.suptitle('ECE & Brier Score by Dataset')

        plot_idx = 0
        for dataset in datasets:
            subset = self.df[self.df['dataset'] == dataset]
            for col, label in metrics_avail:
                ax = axes[plot_idx // n_cols][plot_idx % n_cols]
                plot_idx += 1

                x                = np.arange(len(models_avail))
                width            = 0.8 / max(len(self.STRATEGY_ORDER), 1)
                strategies_avail = [s for s in self.STRATEGY_ORDER
                                    if s in subset['strategy'].values]

                for i, strategy in enumerate(strategies_avail):
                    srows  = subset[subset['strategy'] == strategy]
                    values = [
                        srows[srows['model'] == m][col].mean()
                        if m in srows['model'].values else np.nan
                        for m in models_avail
                    ]
                    offset = (i - len(strategies_avail) / 2 + 0.5) * width
                    bars = ax.bar(x + offset, values, width,
                           label=strategy,
                           color=self.STRATEGY_COLORS.get(strategy, '#888'),
                           edgecolor='white', linewidth=0.5, alpha=0.88)
                    ax.bar_label(bars, labels=[f'{v:.3f}' if not np.isnan(v) else '' for v in values],
                             fontsize=6, padding=2, rotation=0, fontweight='bold', color='#444444')

                ax.set_title(dataset, fontsize=11)
                ax.set_xticks(x)
                ax.set_xticklabels(models_avail, rotation=20, ha='right', fontsize=8)
                ax.set_ylabel(label, fontsize=9)
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, ylims[col])

        for idx in range(plot_idx, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        handles = [plt.Rectangle((0, 0), 1, 1,
                                  color=self.STRATEGY_COLORS.get(s, '#888'), label=s)
                   for s in self.STRATEGY_ORDER]
        fig.legend(handles=handles, loc='lower center', ncol=len(self.STRATEGY_ORDER),
                   bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)

        plt.tight_layout()
        self._save_fig('calibration_comparison.png')
    
    def plot_aggregated_confusion_matrices(self, figsize=(9, 6)) -> None:
        self._check_loaded()

        models  = [m for m in self.MODEL_ORDER if any(r.model == m for r in self.records)]
        classes = self.EMOTION_CLASSES

        for model in models:
            matrices = []
            for rec in self.records:
                if rec.model != model:
                    continue
                cm = rec.metrics.get('confusion_matrix')
                if cm is None:
                    continue
                cm = np.array(cm, dtype=float)
                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                matrices.append(cm / row_sums)

            if not matrices:
                continue

            mean_cm = np.mean(matrices, axis=0)

            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(f'Confusion Matrix | {model}', y=1.01)

            sns.heatmap(mean_cm, ax=ax,
                        annot=True, fmt='.2f', cmap='Blues',
                        vmin=0, vmax=1,
                        linewidths=0.3, linecolor='#dddddd',
                        cbar_kws={'shrink': 0.75, 'label': 'Mean Recall (normalized)'},
                        annot_kws={'size': 9},
                        xticklabels=classes,
                        yticklabels=classes)

            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', rotation=0,  labelsize=9)

            plt.tight_layout()
            self._save_fig(f'aggregated_confusion_matrix_{model.lower()}.png')
            plt.close(fig)