from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .BaseComparisonHandler import BaseComparisonHandler
from DirectoryManager import DirectoryManager


class ComparisonTFLiteHandler(BaseComparisonHandler):
    """Handler for visualizing and comparing TFLite conversion results across all experiments."""

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

    # ── visualizations ────────────────────────────────────────────────────────────

    def plot_size_vs_accuracy(self, figsize=(13, 7)) -> None:
        self._check_loaded()

        variant_markers = {
            'float32':       ('o', 'float32'),
            'dynamic_quant': ('s', 'Dynamic Range'),
            'int8_quant':    ('^', 'Full INT8'),
        }
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Model Size vs Accuracy  —  TFLite (all 156 variants)',
                     fontsize=13, fontweight='bold')
        seen_models, seen_variants = set(), set()
        for _, row in self.df.iterrows():
            model = row['model']
            color = self.MODEL_COLORS.get(model, '#888')
            for vkey, (marker, _) in variant_markers.items():
                acc  = row.get(f'tflite_{vkey}_accuracy')
                size = row.get(f'tflite_{vkey}_size_kb')
                if pd.isna(acc) or pd.isna(size):
                    continue
                ax.scatter(size / 1024, acc, color=color, marker=marker,
                           s=75, alpha=0.82, edgecolors='white', linewidths=0.6, zorder=3)
                seen_models.add(model)
                seen_variants.add(vkey)
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('TFLite Test Accuracy')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        model_h = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.MODEL_COLORS.get(m, '#888'), markersize=9, label=m)
            for m in self.MODEL_ORDER if m in seen_models
        ]
        var_h = [
            plt.Line2D([0], [0], marker=mk, color='w',
                       markerfacecolor='#555', markersize=9, label=lbl)
            for vkey, (mk, lbl) in variant_markers.items() if vkey in seen_variants
        ]
        l1 = ax.legend(handles=model_h, loc='lower right',
                       title='Model', frameon=True, fontsize=9)
        ax.add_artist(l1)
        ax.legend(handles=var_h, loc='lower left',
                  title='Quantization', frameon=True, fontsize=9)
        self._save_fig('size_vs_accuracy.png')

    def plot_quantization_accuracy_delta(self, figsize=(14, 9)) -> None:
        self._check_loaded()
        
        needed = ['model', 'dataset', 'strategy', 'test_accuracy',
                  'tflite_float32_accuracy', 'tflite_dynamic_quant_accuracy',
                  'tflite_int8_quant_accuracy']
        df = self.df[[c for c in needed if c in self.df.columns]].copy()
        if 'test_accuracy' not in df.columns:
            df['test_accuracy'] = df['tflite_float32_accuracy']
        df = df.dropna(subset=['test_accuracy'])
        if df.empty:
            print('  [skip] No data for quantization delta plot')
            return

        for vkey in ('dynamic_quant', 'int8_quant'):
            col = f'tflite_{vkey}_accuracy'
            if col in df.columns:
                df[f'delta_{vkey}'] = df[col] - df['test_accuracy']

        delta_cols = [c for c in ('delta_dynamic_quant', 'delta_int8_quant')
                      if c in df.columns]
        if not delta_cols:
            print('  [skip] No TFLite variant data for delta plot')
            return

        col_labels = {'delta_dynamic_quant': 'Dynamic Range', 'delta_int8_quant': 'Full INT8'}
        pivot_data = df.groupby(['model', 'dataset'])[delta_cols].mean().reset_index()
        keras_acc  = (df.groupby(['model', 'dataset'])['test_accuracy']
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
        fig.suptitle(
            'Quantization Accuracy Delta  —  Δ = TFLite accuracy − Keras accuracy\n'
            '(negative = accuracy loss after quantization; Keras baseline in parentheses)',
            fontsize=12, fontweight='bold',
        )
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
                        cbar_kws={'label': 'Δ Accuracy', 'shrink': 0.75},
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
        self._save_fig('quantization_accuracy_delta.png')

    def plot_tflite_aggregated_scatter(self, figsize=(12, 7)) -> None:
        self._check_loaded()

        variant_markers = {
            'float32':       ('o', 'float32',       '#cccccc'),
            'dynamic_quant': ('s', 'Dynamic Range', '#888888'),
            'int8_quant':    ('^', 'Full INT8',      '#333333'),
        }
        abbrev = {
            'SimpleCNN': 'SC', 'VGG16': 'VGG', 'ResNet50': 'RN50',
            'MobileNetV2': 'MNV2', 'EfficientNetB0': 'ENB0',
        }

        rows = []
        for model in self.MODEL_ORDER:
            mdf = self.df[self.df['model'] == model]
            for vkey, (marker, label, _) in variant_markers.items():
                mean_acc  = mdf[f'tflite_{vkey}_accuracy'].mean(skipna=True)
                mean_size = mdf[f'tflite_{vkey}_size_kb'].mean(skipna=True) / 1024
                lat_col   = f'tflite_{vkey}_p95_ms'
                mean_lat  = mdf[lat_col].mean(skipna=True) if lat_col in mdf.columns else 5.0
                if pd.isna(mean_acc) or pd.isna(mean_size):
                    continue
                rows.append({'model': model, 'variant': vkey, 'label': label,
                             'accuracy': mean_acc, 'size_mb': mean_size, 'p95_ms': mean_lat})
        if not rows:
            print('  [skip] No TFLite data for aggregated scatter')
            return

        agg          = pd.DataFrame(rows)
        lat_vals     = agg['p95_ms'].values
        lat_min, lat_max = lat_vals.min(), lat_vals.max()
        lat_range    = lat_max - lat_min if lat_max > lat_min else 1.0
        marker_sizes = 50 + 350 * (lat_vals - lat_min) / lat_range

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(
            'TFLite: Model Size vs Accuracy  —  Aggregated (mean across datasets & strategies)\n'
            'Point size = mean p95 inference latency  |  5 models × 3 quantization variants',
            fontsize=11, fontweight='bold',
        )
        for i, row in agg.iterrows():
            mk    = variant_markers[row['variant']][0]
            color = self.MODEL_COLORS.get(row['model'], '#888')
            ax.scatter(row['size_mb'], row['accuracy'], color=color, marker=mk,
                       s=marker_sizes[i], alpha=0.88,
                       edgecolors='white', linewidths=0.8, zorder=4)
            ax.annotate(abbrev.get(row['model'], row['model']),
                        (row['size_mb'], row['accuracy']),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=7.5, color=color, fontweight='bold')

        for vkey, (mk, vlabel, frontier_color) in variant_markers.items():
            vdata = agg[agg['variant'] == vkey].sort_values('size_mb')
            if len(vdata) < 2:
                continue
            pareto, best_acc = [], -1.0
            for _, r in vdata.iterrows():
                if r['accuracy'] >= best_acc:
                    pareto.append(r)
                    best_acc = r['accuracy']
            if len(pareto) >= 2:
                ax.plot([p['size_mb'] for p in pareto],
                        [p['accuracy'] for p in pareto],
                        color=frontier_color, linewidth=1.0,
                        linestyle='--', alpha=0.5, zorder=2)

        ax.set_xlabel('Mean Model Size (MB)')
        ax.set_ylabel('Mean TFLite Accuracy')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        model_h = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.MODEL_COLORS.get(m, '#888'), markersize=9, label=m)
            for m in self.MODEL_ORDER if m in agg['model'].values
        ]
        var_h = [
            plt.Line2D([0], [0], marker=mk, color='w',
                       markerfacecolor='#555', markersize=9, label=lbl)
            for vkey, (mk, lbl, _) in variant_markers.items()
            if vkey in agg['variant'].values
        ]
        for lat_val, s_val in [(lat_min, 50), (lat_max, 400)]:
            model_h.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#aaa',
                                      markersize=(s_val / 50) ** 0.5 * 4,
                                      label=f'p95 latency ~{lat_val:.1f} ms'))
        l1 = ax.legend(handles=model_h, loc='lower right',
                       title='Model / Latency', frameon=True, fontsize=8)
        ax.add_artist(l1)
        ax.legend(handles=var_h, loc='upper left',
                  title='Quantization', frameon=True, fontsize=8)
        self._save_fig('tflite_aggregated_scatter.png')