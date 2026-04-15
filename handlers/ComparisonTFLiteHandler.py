import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .BaseComparisonHandler import BaseComparisonHandler


class ComparisonTFLiteHandler(BaseComparisonHandler):
    """Handler for loading, processing and visualizing TFLite conversion experiment results in the comparison experiment context."""

    def __init__(
        self,
        train_experiments_dir: str,
        visualizations_directory: str,
    ):
        super().__init__(
            train_experiments_dir=train_experiments_dir,
            visualizations_directory=visualizations_directory,
        )
        print('ComparisonTFLiteHandler initialized.')

    # ── visualization ────────────────────────────────────────────────────────

    def plot_size_vs_accuracy_by_strategy(self, figsize=(14, 10)) -> None:
        """Plot scatter charts of TFLite model size vs test accuracy, colored by model and faceted by strategy."""
        self._check_loaded()

        strategies = [s for s in self.STRATEGY_ORDER if s in self.df['strategy'].values]
        n_cols = 2
        n_rows = int(np.ceil(len(strategies) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharey=True)
        fig.suptitle('Model Size vs TFLite Accuracy by Strategy', y=1.01)

        for idx, strategy in enumerate(strategies):
            ax     = axes[idx // n_cols][idx % n_cols]
            subset = self.df[self.df['strategy'] == strategy]

            for _, row in subset.iterrows():
                model = row['model']
                color = self.MODEL_COLORS.get(model, '#888')
                for vkey in self.TFLITE_VARIANTS:
                    acc  = row.get(f'tflite_{vkey}_accuracy')
                    size = row.get(f'tflite_{vkey}_size_kb')
                    if pd.isna(acc) or pd.isna(size):
                        continue
                    ax.scatter(size / 1024, acc, color=color,
                               s=60, alpha=0.8, edgecolors='white',
                               linewidths=0.6, zorder=3)

            ax.set_title(strategy, fontsize=11)
            ax.set_xlabel('Model Size (MB)', fontsize=9)
            ax.set_ylabel('TFLite Test Accuracy' if idx % n_cols == 0 else '', fontsize=9)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.grid(axis='both', alpha=0.3)

        for idx in range(len(strategies), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        model_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.MODEL_COLORS.get(m, '#888'),
                       markersize=8, label=m)
            for m in self.MODEL_ORDER if m in self.df['model'].values
        ]
        fig.legend(handles=model_handles, loc='lower center',
                   ncol=len(model_handles), bbox_to_anchor=(0.5, -0.02),
                   frameon=False, fontsize=10)

        plt.tight_layout()
        for row_axes in axes:
            for ax in row_axes:
                ax.tick_params(labelleft=True)
        plt.subplots_adjust(bottom=0.10)
        self._save_fig('size_vs_accuracy_by_strategy.png')

    def plot_quantization_accuracy_delta(self, figsize=(14, 8)) -> None:
        """Plot heatmaps of TFLite quantization accuracy delta vs Keras baseline across models and datasets, faceted by quantization variant."""
        self._check_loaded()

        needed = ['model', 'dataset', 'strategy', 'test_accuracy',
                  'tflite_float32_accuracy', 'tflite_dynamic_quant_accuracy',
                  'tflite_int8_quant_accuracy']
        df = self.df[[c for c in needed if c in self.df.columns]].copy()
        if 'test_accuracy' not in df.columns:
            df['test_accuracy'] = df['tflite_float32_accuracy']
        df = df.dropna(subset=['test_accuracy'])
        if df.empty:
            print('No data for quantization delta plot')
            return

        for vkey in ('dynamic_quant', 'int8_quant'):
            col = f'tflite_{vkey}_accuracy'
            if col in df.columns:
                df[f'delta_{vkey}'] = df[col] - df['test_accuracy']

        delta_cols = [c for c in ('delta_dynamic_quant', 'delta_int8_quant')
                      if c in df.columns]
        if not delta_cols:
            print('No TFLite variant data for delta plot')
            return

        col_labels = {'delta_dynamic_quant': 'Dynamic Range', 'delta_int8_quant': 'Full INT8'}
        pivot_data = df.groupby(['model', 'dataset'], observed=True)[delta_cols].mean().reset_index()
        keras_acc  = (df.groupby(['model', 'dataset'], observed=True)['test_accuracy']
                        .mean().reset_index()
                        .rename(columns={'test_accuracy': 'keras_acc'}))
        pivot_data = pivot_data.merge(keras_acc, on=['model', 'dataset'])

        pivots = {}
        for col in delta_cols:
            p         = pivot_data.pivot(index='model', columns='dataset', values=col)
            row_order = [m for m in self.MODEL_ORDER if m in p.index]
            col_order = [d for d in self.DATASET_ORDER if d in p.columns]
            pivots[col] = p.reindex(row_order)[col_order]

        keras_pivot = pivot_data.pivot(index='model', columns='dataset', values='keras_acc')
        keras_pivot = keras_pivot.reindex(
            [m for m in self.MODEL_ORDER if m in keras_pivot.index])
        keras_pivot = keras_pivot[
            [d for d in self.DATASET_ORDER if d in keras_pivot.columns]]

        n_variants = len(delta_cols)
        fig, axes  = plt.subplots(1, n_variants, figsize=figsize, sharey=True)
        if n_variants == 1:
            axes = [axes]
        fig.suptitle('TFLite Quantization Accuracy Delta', y=1.01)
        abs_max = max(pivots[c].abs().max().max() for c in delta_cols
                      if not pivots[c].empty)
        abs_max = max(abs_max, 0.01)

        for ax, col in zip(axes, delta_cols):
            pivot = pivots[col]
            mask  = pivot.isna()
            annot = np.full(pivot.shape, '', dtype=object)
            for ri, ridx in enumerate(pivot.index):
                for ci, cidx in enumerate(pivot.columns):
                    dval = pivot.loc[ridx, cidx]
                    base = (keras_pivot.loc[ridx, cidx]
                            if ridx in keras_pivot.index else np.nan)
                    if pd.notna(dval):
                        sign     = '+' if dval >= 0 else ''
                        base_str = f'\n({base:.2f})' if pd.notna(base) else ''
                        annot[ri, ci] = f'{sign}{dval:.3f}{base_str}'
            sns.heatmap(pivot, ax=ax, mask=mask,
                        annot=annot, fmt='', cmap='RdBu', center=0,
                        vmin=-abs_max, vmax=abs_max,
                        linewidths=0.5, linecolor='#dddddd',
                        cbar_kws={'label': 'Accuracy Delta', 'shrink': 0.75},
                        annot_kws={'size': 8})
            if mask.any().any():
                sns.heatmap(pivot, ax=ax, mask=~mask,
                            annot=False, cmap=['#eeeeee'], vmin=0, vmax=1,
                            linewidths=0.5, linecolor='#dddddd', cbar=False)
            ax.set_title(col_labels[col], fontsize=11)
            ax.set_xlabel('Dataset', labelpad=8)
            ax.set_ylabel('Model' if ax is axes[0] else '', labelpad=8)
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        for ax in axes:
            ax.tick_params(labelleft=True)
        self._save_fig('quantization_accuracy_delta.png')

    def plot_accuracy_heatmap(self, figsize=(9, 6)) -> None:
        """Plot a heatmap of TFLite accuracy across models and quantization variants, faceted by dataset and strategy."""
        self._check_loaded()

        variant_labels = {
            'float32':       'float32',
            'dynamic_quant': 'Dynamic Range',
            'int8_quant':    'Full INT8',
        }
        models = [m for m in self.MODEL_ORDER if m in self.df['model'].values]

        pivot_acc  = pd.DataFrame(index=models, columns=self.TFLITE_VARIANTS, dtype=float)
        for model in models:
            mdf = self.df[self.df['model'] == model]
            for vkey in self.TFLITE_VARIANTS:
                pivot_acc.loc[model, vkey] = mdf[f'tflite_{vkey}_accuracy'].mean(skipna=True)

        pivot_acc  = pivot_acc.rename(columns=variant_labels)
        annot      = pivot_acc.map(lambda v: f'{v:.1%}' if pd.notna(v) else '')

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('TFLite Mean Accuracy | Model - TFLite Variant', y=1.01)

        sns.heatmap(
            pivot_acc.astype(float),
            ax=ax,
            annot=annot,
            fmt='',
            cmap='YlGn',
            vmin=pivot_acc.astype(float).min().min(),
            vmax=pivot_acc.astype(float).max().max(),
            linewidths=0.5,
            linecolor='#dddddd',
            cbar_kws={'label': 'Mean Accuracy', 'shrink': 0.75},
            annot_kws={'size': 10},
        )
        ax.set_xlabel('TFLite Variant', fontsize=10)
        ax.set_ylabel('Model', fontsize=10)
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

        plt.tight_layout()
        self._save_fig('tflite_accuracy_heatmap.png')

    def plot_compression_ratio_scatter(self, figsize=(10, 8)) -> None:
        """Plot scatter charts of TFLite compression ratio vs accuracy and accuracy delta, colored by model and faceted by quantization variant."""
        self._check_loaded()

        cr_cols = [f'tflite_{v}_compression_ratio' for v in self.TFLITE_VARIANTS]
        ad_cols = [f'tflite_{v}_accuracy_delta'    for v in self.TFLITE_VARIANTS]

        has_cr = any(c in self.df.columns and self.df[c].notna().any() for c in cr_cols)
        has_ad = any(c in self.df.columns and self.df[c].notna().any() for c in ad_cols)

        if not has_cr:
            print('No compression_ratio data available.')
            return

        variant_labels = {
            'float32':       'float32',
            'dynamic_quant': 'Dynamic Range',
            'int8_quant':    'Full INT8',
        }

        rows = []
        for _, row in self.df.iterrows():
            model = str(row['model'])
            for vkey in self.TFLITE_VARIANTS:
                cr  = row.get(f'tflite_{vkey}_compression_ratio')
                acc = row.get(f'tflite_{vkey}_accuracy')
                if has_ad:
                    delta = row.get(f'tflite_{vkey}_accuracy_delta')
                else:
                    keras_acc = row.get('test_accuracy')
                    delta = (float(acc) - float(keras_acc)
                            if pd.notna(acc) and pd.notna(keras_acc) else None)
                if pd.isna(cr) or pd.isna(acc):
                    continue
                rows.append({
                    'model':   model,
                    'variant': vkey,
                    'dataset': str(row['dataset']),
                    'cr':      float(cr),
                    'acc':     float(acc),
                    'delta':   float(delta) if delta is not None and not pd.isna(delta) else 0.0,
                })

        if not rows:
            print('No data for compression-accuracy Pareto plot.')
            return

        df_plot  = pd.DataFrame(rows)
        variants = [v for v in self.TFLITE_VARIANTS if v in df_plot['variant'].values]

        model_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=self.MODEL_COLORS.get(m, '#888'),
                    markersize=8, label=m)
            for m in self.MODEL_ORDER if m in df_plot['model'].values
        ]

        for vkey in variants:
            vdf = df_plot[df_plot['variant'] == vkey]
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(f'TFLite Compression Ratio vs Accuracy | {variant_labels[vkey]}', y=1.01)

            for ax, y_col, y_label in zip(
                axes,
                ['acc',   'delta'],
                ['TFLite Accuracy', 'Accuracy Delta vs Keras'],
            ):
                for _, row in vdf.iterrows():
                    color = self.MODEL_COLORS.get(row['model'], '#888')
                    ax.scatter(row['cr'], row[y_col], color=color,
                            s=55, alpha=0.72, edgecolors='white',
                            linewidths=0.5, zorder=3)

                if y_col == 'delta':
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='-', alpha=0.4)

                ax.set_xlabel('Compression Ratio (Keras / TFLite)', fontsize=9)
                ax.set_title(y_label, fontsize=11)
                if y_col == 'acc':
                    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
                ax.grid(axis='both', alpha=0.3)

            fig.legend(handles=model_handles, loc='lower center',
                    ncol=len(model_handles), bbox_to_anchor=(0.5, -0.04),
                    frameon=False, fontsize=9)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12)
            self._save_fig(f'compression_ratio_scatter_{vkey}.png')