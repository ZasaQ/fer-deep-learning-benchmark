from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
import cv2
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from scipy.stats import pearsonr

from .base_handler import BaseHandler
from .dataset_handler import DatasetHandler
from .data_augmentation_handler import DataAugmentationHandler


class EvaluationHandler(BaseHandler):
    """Handles model evaluation, metrics computation and performance visualization."""

    def __init__(self,
                 model: tf.keras.Model,
                 data_augmentation_handler: DataAugmentationHandler,
                 dataset_handler: DatasetHandler,
                 epoch_class_f1: Optional[List[Dict[str, float]]],
                 visualizations_directory: str):
        self.model = model
        self.data_augmentation_handler = data_augmentation_handler
        self.dataset_handler = dataset_handler
        super().__init__(visualizations_directory)

        self.report: Optional[dict] = None
        self.confusion_matrix: Optional[np.ndarray] = None
        self.per_class_acc: Optional[np.ndarray] = None
        self.test_loss: Optional[float] = None
        self.test_accuracy: Optional[float] = None
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        self.y_pred_proba: Optional[np.ndarray] = None

        self.epoch_class_f1: Optional[List[Dict[str, float]]] = epoch_class_f1

        print('EvaluationHandler has been initialized.')

    # ── private helpers ─────────────────────────────────────

    def _get_classification_report(self) -> dict:
        """Return sklearn classification report as a dict."""
        return classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.dataset_handler.class_labels,
            output_dict=True,
            zero_division=0
        )

    def _get_top_misclassification_pairs(self, top_n: int):
        """Return the top_n most frequent true→predicted misclassification pairs."""
        if self.y_true is None or self.y_pred is None:
            return None

        cm     = self.confusion_matrix
        labels = self.dataset_handler.class_labels
        pairs  = []

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        'true':        labels[i],
                        'predicted':   labels[j],
                        'count':       int(cm[i, j]),
                        'pct_of_true': cm[i, j] / cm[i].sum() * 100
                    })

        pairs.sort(key=lambda x: x['count'], reverse=True)
        return pairs[:top_n]

    def _load_images_by_indices(self, indices: np.ndarray) -> dict:
        """Load specific images from the test generator by index without loading all into RAM."""
        test_gen = self.data_augmentation_handler.test_generator
        test_gen.reset()

        needed  = set(indices.tolist())
        images  = {}
        current = 0

        for batch_x, _ in test_gen:
            for img in batch_x:
                if current in needed:
                    images[current] = img
                current += 1
                if len(images) == len(needed):
                    return images
            if current > max(needed):
                break

        return images

    # ── evaluation ──────────────────────────────────────────

    def evaluate(self, split: str) -> Tuple[float, float]:
        """Evaluate model on val or test split and return (loss, accuracy)."""
        generator_map = {
            'val':  self.data_augmentation_handler.val_generator,
            'test': self.data_augmentation_handler.test_generator,
        }

        generator = generator_map.get(split)
        if generator is None:
            raise ValueError(f"Invalid split '{split}'. Choose 'val' or 'test'.")

        print(f"Evaluating on {split} set...")
        loss, accuracy = self.model.evaluate(generator, verbose=1)

        if split == 'test':
            self.test_loss     = loss
            self.test_accuracy = accuracy

        print(f"\n{split.capitalize()} Results:")
        print(f"  Loss:     {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        return loss, accuracy

    def predict(self, split: str) -> None:
        """Run inference and store y_true, y_pred, y_pred_proba for visualization."""
        generator_map = {
            'val':  self.data_augmentation_handler.val_generator,
            'test': self.data_augmentation_handler.test_generator,
        }

        generator = generator_map.get(split)
        if generator is None:
            raise ValueError(f"Invalid split '{split}'. Choose 'val' or 'test'.")

        print(f"Generating predictions on {split} set...")
        generator.reset()
        self.y_pred_proba     = self.model.predict(generator, verbose=1)
        self.y_pred           = np.argmax(self.y_pred_proba, axis=1)
        self.y_true           = generator.classes
        self.report           = self._get_classification_report()
        self.confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.per_class_acc    = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)

        print(f"Predictions complete: {len(self.y_pred)} samples")

    # ── visualizations ──────────────────────────────────────

    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (18, 7)) -> None:
        """Raw count and row-normalized confusion matrices side by side."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        cm      = confusion_matrix(self.y_true, self.y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        row_sums = cm.sum(axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.dataset_handler.class_labels,
            yticklabels=self.dataset_handler.class_labels,
            ax=ax1, vmin=0, vmax=row_sums.max(),
            cbar_kws={'label': 'Count'},
            square=True, linewidths=0.5, linecolor='white',
        )
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        ax1.set_title('Raw Counts')

        sns.heatmap(
            cm_norm, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=self.dataset_handler.class_labels,
            yticklabels=self.dataset_handler.class_labels,
            ax=ax2, vmin=0, vmax=100,
            cbar_kws={'label': '% of True Class'},
            square=True, linewidths=0.5, linecolor='white',
        )
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title('Normalized (%)')

        plt.suptitle('Confusion Matrices')
        self._save_fig('confusion_matrix.png')

    def plot_classification_report(self, figsize: Tuple[int, int] = (8, 8)) -> None:
        """Heatmap of precision, recall and F1-score per class."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        report  = self._get_classification_report()
        metrics = ['precision', 'recall', 'f1-score']
        data    = np.array([
            [report[label][metric] for metric in metrics]
            for label in self.dataset_handler.class_labels
        ])

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=metrics,
            yticklabels=self.dataset_handler.class_labels,
            ax=ax, vmin=0, vmax=1
        )
        ax.set_title('Classification Report')

        self._save_fig('classification_report.png')

    def plot_precision_recall_f1(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """Grouped bar chart of precision, recall and F1-score per class."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        report    = self._get_classification_report()
        precision = [report[label]['precision'] for label in self.dataset_handler.class_labels]
        recall    = [report[label]['recall']    for label in self.dataset_handler.class_labels]
        f1_score  = [report[label]['f1-score']  for label in self.dataset_handler.class_labels]

        fig, ax = plt.subplots(figsize=figsize)
        x     = np.arange(len(self.dataset_handler.class_labels))
        width = 0.25

        bars_p = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        bars_r = ax.bar(x,         recall,    width, label='Recall',    color='#e74c3c', alpha=0.8)
        bars_f = ax.bar(x + width, f1_score,  width, label='F1-Score',  color='#2ecc71', alpha=0.8)

        for bars in [bars_p, bars_r, bars_f]:
            ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)

        overall_mean = np.mean([np.mean(precision), np.mean(recall), np.mean(f1_score)])
        ax.axhline(y=overall_mean, color='black', linestyle='--', alpha=0.7, label='Overall mean')
        ax.text(len(self.dataset_handler.class_labels) - 0.5, overall_mean + 0.01,
                f'Overall mean: {overall_mean:.3f}', color='black', fontsize=8,
                va='bottom', ha='right', fontweight='bold')

        ax.set_ylabel('Score')
        ax.set_title(f'Precision, Recall & F1-Score | {CONFIG["dataset"]} | {CONFIG["model"]}')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dataset_handler.class_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        self._save_fig('metrics_bar_chart.png')

    def plot_f1_ranking(self, figsize: Tuple[int, int] = (8, 6)) -> None:
        """Horizontal bar chart of per-class F1-score ranked from worst to best."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        report    = self._get_classification_report()
        f1_scores = np.array([report[label]['f1-score'] for label in self.dataset_handler.class_labels])
        macro_f1  = report['macro avg']['f1-score']

        sorted_indices = np.argsort(f1_scores)
        sorted_labels  = [self.dataset_handler.class_labels[i] for i in sorted_indices]
        sorted_f1      = f1_scores[sorted_indices]
        colors = ['#e74c3c' if x < 0.5 else '#f39c12' if x < 0.7 else '#2ecc71' for x in sorted_f1]

        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(sorted_labels)), sorted_f1, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel('F1-Score')
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', alpha=0.3)

        ax.axvline(x=macro_f1, color='black', linestyle='--', alpha=0.7, label='Macro F1')
        ax.text(macro_f1 + 0.02, 0, f'Macro F1: {macro_f1:.3f}',
                color='black', fontweight='bold', fontsize=8, va='bottom')

        for i, v in enumerate(sorted_f1):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.8, label='F1 < 0.5'),
            Patch(facecolor='#f39c12', alpha=0.8, label='0.5 <= F1 < 0.7'),
            Patch(facecolor='#2ecc71', alpha=0.8, label='F1 >= 0.7'),
            plt.Line2D([0], [0], color='black', linestyle='--', label='Macro F1'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.suptitle('F1-Score Ranking')
        self._save_fig('f1_ranking.png')

    def plot_per_class_accuracy(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Bar chart of per-class accuracy with overall and random chance baselines."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        per_class_acc = self.per_class_acc
        class_counts  = self.confusion_matrix.sum(axis=1)
        colors = ['#e74c3c' if x < 0.5 else '#f39c12' if x < 0.7 else '#2ecc71'
                  for x in per_class_acc]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(range(len(per_class_acc)), per_class_acc, color=colors, alpha=0.8)

        for bar, acc in zip(bars, per_class_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        overall_acc = np.mean(per_class_acc)
        random_acc  = 1 / len(self.dataset_handler.class_labels)

        ax.axhline(overall_acc, color='black', linestyle='--', alpha=0.7)
        ax.text(len(per_class_acc) - 0.5, overall_acc + 0.01,
                f'Overall: {overall_acc:.3f}', color='black', fontweight='bold', fontsize=9, ha='right')

        ax.axhline(random_acc, color='black', linestyle=':', alpha=0.7)
        ax.text(len(per_class_acc) - 0.5, random_acc + 0.01,
                f'Random: {random_acc:.3f}', color='black', fontweight='bold', fontsize=9, ha='right')

        ax.set_xticks(range(len(self.dataset_handler.class_labels)))
        ax.set_xticklabels(
            [f'{l}\n(n={c:,})' for l, c in zip(self.dataset_handler.class_labels, class_counts)],
            rotation=45, ha='right'
        )
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.8, label='Accuracy < 0.5'),
            Patch(facecolor='#f39c12', alpha=0.8, label='0.5 <= Accuracy < 0.7'),
            Patch(facecolor='#2ecc71', alpha=0.8, label='Accuracy >= 0.7'),
            plt.Line2D([0], [0], color='black',    linestyle='--', label='Overall'),
            plt.Line2D([0], [0], color='darkgray', linestyle=':',  label='Random'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        self._save_fig('per_class_accuracy.png')

    def plot_precision_recall_scatter(self, figsize: Tuple[int, int] = (8, 8)) -> None:
        """Precision vs Recall scatter with F1 iso-curves and color-coded F1 score."""
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        report    = self._get_classification_report()
        precision = np.array([report[label]['precision'] for label in self.dataset_handler.class_labels])
        recall    = np.array([report[label]['recall']    for label in self.dataset_handler.class_labels])
        f1_score  = np.array([report[label]['f1-score']  for label in self.dataset_handler.class_labels])

        fig, ax = plt.subplots(figsize=figsize)

        for f1 in [0.3, 0.5, 0.7, 0.9]:
            r = np.linspace(0.01, 1, 300)
            p = f1 * r / (2 * r - f1)
            p = np.where((p > 0) & (p <= 1), p, np.nan)
            ax.plot(r, p, '--', alpha=0.2, color='gray')
            ax.annotate(f'F1={f1}', xy=(0.9, f1 * 0.9 / (2 * 0.9 - f1)), fontsize=8, color='gray')

        scatter = ax.scatter(recall, precision, s=300, alpha=0.7, c=f1_score, cmap='RdYlGn',
                             edgecolors='black', linewidths=2.5, zorder=3)

        for i, label in enumerate(self.dataset_handler.class_labels):
            ax.annotate(label, (recall[i], precision[i]), xytext=(8, 8),
                        textcoords='offset points', fontweight='bold', fontsize=9)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall Trade-off')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.colorbar(scatter, ax=ax, pad=0.02).set_label('F1-Score', fontweight='bold')

        self._save_fig('precision_recall_scatter.png')

    def plot_sharpness_vs_accuracy(self,
                                   split: str,
                                   num_samples: int,
                                   figsize: Tuple[int, int] = (10, 7)) -> None:
        """
        Scatter plot of per-class mean sharpness (Laplacian variance) vs per-class accuracy.
        """
        if not self._guard(self.y_true is not None and self.confusion_matrix is not None,
                           "No predictions available. Call predict() first."):
            return

        per_class_acc     = self.per_class_acc
        folder            = self.dataset_handler._folder(split)
        samples_per_class = max(1, num_samples // self.dataset_handler.class_num)
        class_sharpness   = {}

        print(f"Computing sharpness ({num_samples} samples from '{split}')...")
        for class_name in self.dataset_handler.class_names:
            class_path = os.path.join(folder, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            vals = []
            for img_name in random.sample(images, min(samples_per_class, len(images))):
                img = Image.open(os.path.join(class_path, img_name)).convert('L')
                arr = np.array(img, dtype=np.float64)
                vals.append(cv2.Laplacian(arr.astype(np.uint8), cv2.CV_64F).var())
            class_sharpness[class_name] = vals

        mean_sharpness = np.array([np.mean(class_sharpness[cn]) for cn in self.dataset_handler.class_names])
        q1_sharpness   = np.array([np.percentile(class_sharpness[cn], 25) for cn in self.dataset_handler.class_names])
        q3_sharpness   = np.array([np.percentile(class_sharpness[cn], 75) for cn in self.dataset_handler.class_names])
        xerr           = np.array([mean_sharpness - q1_sharpness, q3_sharpness - mean_sharpness])
        colors         = plt.cm.tab10(np.linspace(0, 1, self.dataset_handler.class_num))

        r, p_val = pearsonr(mean_sharpness, per_class_acc)

        coeffs = np.polyfit(mean_sharpness, per_class_acc, 1)
        x_line = np.linspace(max(0, mean_sharpness.min() * 0.8),
                             mean_sharpness.max() * 1.1, 200)
        y_line = np.polyval(coeffs, x_line)

        fig, ax = plt.subplots(figsize=figsize)

        for i, label in enumerate(self.dataset_handler.class_labels):
            ax.errorbar(
                mean_sharpness[i], per_class_acc[i],
                xerr=[[xerr[0, i]], [xerr[1, i]]],
                fmt='o', color=colors[i], markersize=10,
                ecolor=colors[i], elinewidth=1.2, capsize=4,
                alpha=0.85, zorder=3,
            )
            ax.annotate(
                label,
                xy=(mean_sharpness[i], per_class_acc[i]),
                xytext=(8, 6), textcoords='offset points',
                fontsize=9, fontweight='bold', color=colors[i],
            )

        ax.plot(x_line, y_line, color='gray', linestyle='--',
                linewidth=1.5, alpha=0.7, zorder=2, label='Regression line')

        p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
        ax.annotate(
            f'r = {r:.3f}\n{p_str}',
            xy=(0.97, 0.05), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85),
        )

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Mean Sharpness — Laplacian Variance')
        ax.set_ylabel('Per-Class Accuracy (test)')
        ax.set_title(
            f'Image Sharpness vs Model Accuracy per Class | {split}\n'
            f'Sampling {num_samples} images'
        )
        ax.legend(loc='upper left')
        ax.grid(axis='both', alpha=0.3)

        self._save_fig('sharpness_vs_accuracy.png')

    def plot_roc_curves(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Per-class ROC curves sorted by AUC, with macro average overlay."""
        if self.y_true is None or self.y_pred_proba is None:
            print("No predictions available. Call predict() first.")
            return

        y_true_bin = label_binarize(self.y_true, classes=range(self.dataset_handler.class_num))

        aucs = []
        for i in range(self.dataset_handler.class_num):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_pred_proba[:, i])
            aucs.append(auc(fpr, tpr))

        sorted_indices = np.argsort(aucs)[::-1]

        fig, ax = plt.subplots(figsize=figsize)

        for i in sorted_indices:
            class_label = self.dataset_handler.class_labels[i]
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_pred_proba[:, i])
            ax.plot(fpr, tpr, linewidth=2, label=f'{class_label} (AUC = {aucs[i]:.3f})')

        macro_auc               = roc_auc_score(y_true_bin, self.y_pred_proba, average='macro')
        fpr_macro, tpr_macro, _ = roc_curve(y_true_bin.ravel(), self.y_pred_proba.ravel())
        ax.plot(fpr_macro, tpr_macro, 'k-', linewidth=2.5,
                label=f'Macro avg (AUC = {macro_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(axis='both', alpha=0.3)

        self._save_fig('roc_curves.png')

    def plot_confidence_distribution(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """Histogram of prediction confidence for correct vs incorrect predictions."""
        if not self._guard(self.y_pred_proba is not None,
                       "No predictions available. Call predict() first."):
            return

        confidence     = np.max(self.y_pred_proba, axis=1)
        correct        = (self.y_pred == self.y_true)
        conf_correct   = confidence[correct]
        conf_incorrect = confidence[~correct]

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(conf_correct,   bins=50, alpha=0.7, color='green', label='Correct')
        ax.hist(conf_incorrect, bins=50, alpha=0.7, color='red',   label='Incorrect')

        ax.axvline(conf_correct.mean(),   color='darkgreen', linestyle='--')
        ax.axvline(conf_incorrect.mean(), color='darkred',   linestyle='--')

        ymax = ax.get_ylim()[1]
        ax.text(conf_correct.mean(),   ymax, f' mean\n {conf_correct.mean():.3f}',
                color='darkgreen', fontweight='bold', fontsize=8, va='top')
        ax.text(conf_incorrect.mean(), ymax, f' mean\n {conf_incorrect.mean():.3f}',
                color='darkred', fontweight='bold', fontsize=8, va='top')

        ax.text(0.02, 0.97,
                f'Correct\nn={len(conf_correct):,}, med={np.median(conf_correct):.3f}, mean={conf_correct.mean():.3f}',
                transform=ax.transAxes, color='darkgreen', fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', edgecolor='darkgreen', alpha=0.8))
        ax.text(0.02, 0.87,
                f'Incorrect\nn={len(conf_incorrect):,}, med={np.median(conf_incorrect):.3f}, mean={conf_incorrect.mean():.3f}',
                transform=ax.transAxes, color='darkred', fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', edgecolor='darkred', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.title('Confidence Distribution')
        self._save_fig('confidence_distribution.png')

    def plot_top_correct_classified(self, top_n: int, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Image grid of the most confidently correct predictions with per-class probability bars."""
        if not self._guard(self.y_true is not None and self.y_pred_proba is not None,
                    "No predictions available. Call predict() first."):
            return

        correct = np.where(self.y_pred == self.y_true)[0]
        if len(correct) == 0:
            return

        confidence     = np.max(self.y_pred_proba[correct], axis=1)
        sorted_indices = correct[np.argsort(-confidence)][:top_n]
        images         = self._load_images_by_indices(sorted_indices)

        n_cols = min(len(sorted_indices), 4)
        n_rows = int(np.ceil(len(sorted_indices) / n_cols))
        fig    = plt.figure(figsize=(figsize[0], n_rows * 3.5))

        for plot_idx, sample_idx in enumerate(sorted_indices):
            img    = images[sample_idx]
            true_l = self.dataset_handler.class_labels[self.y_true[sample_idx]]
            proba  = self.y_pred_proba[sample_idx]
            conf   = np.max(proba)
            color  = '#27ae60' if conf > 0.9 else '#2ecc71' if conf > 0.7 else '#a9dfbf'
            labels = self.dataset_handler.class_labels

            row = plot_idx // n_cols
            col = plot_idx  % n_cols

            ax_img = fig.add_subplot(n_rows, n_cols * 2, row * n_cols * 2 + col * 2 + 1)
            if self.dataset_handler.channels == 1:
                ax_img.imshow(
                    img.reshape(self.dataset_handler.rows, self.dataset_handler.cols),
                    cmap='gray'
                )
            else:
                ax_img.imshow(img)
            ax_img.set_title(f'{true_l}\n({conf:.2f})', color=color, fontsize=8, fontweight='bold')
            ax_img.axis('off')

            ax_bar = fig.add_subplot(n_rows, n_cols * 2, row * n_cols * 2 + col * 2 + 2)
            bar_colors = [color if lbl == true_l else '#bdc3c7' for lbl in labels]
            ax_bar.barh(labels, proba, color=bar_colors, height=0.6)
            ax_bar.set_xlim(0, 1.15)
            ax_bar.tick_params(axis='y', labelsize=6)
            ax_bar.tick_params(axis='x', labelsize=6)
            ax_bar.set_xlabel('P', fontsize=6)
            ax_bar.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            for i, (lbl, p) in enumerate(zip(labels, proba)):
                txt_color = color if lbl == true_l else '#7f8c8d'
                ax_bar.text(p + 0.02, i, f'{p:.2f}', va='center', fontsize=6,
                            color=txt_color, fontweight='bold' if lbl == true_l else 'normal')

        plt.suptitle(f'Top {top_n} Confident Correct Predictions', y=1.01, fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig('top_correct_classified.png')

    def plot_top_misclassified(self, top_n: int, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Image grid of the most confidently misclassified samples with per-class probability bars."""
        if not self._guard(self.y_true is not None and self.y_pred_proba is not None,
                    "No predictions available. Call predict() first."):
            return

        misclassified = np.where(self.y_pred != self.y_true)[0]
        if len(misclassified) == 0:
            return

        confidence     = np.max(self.y_pred_proba[misclassified], axis=1)
        sorted_indices = misclassified[np.argsort(-confidence)][:top_n]
        images         = self._load_images_by_indices(sorted_indices)

        n_cols = min(len(sorted_indices), 4)
        n_rows = int(np.ceil(len(sorted_indices) / n_cols))
        fig    = plt.figure(figsize=(figsize[0], n_rows * 3.5))

        for plot_idx, sample_idx in enumerate(sorted_indices):
            img    = images[sample_idx]
            true_l = self.dataset_handler.class_labels[self.y_true[sample_idx]]
            pred_l = self.dataset_handler.class_labels[self.y_pred[sample_idx]]
            proba  = self.y_pred_proba[sample_idx]
            conf   = np.max(proba)
            labels = self.dataset_handler.class_labels

            row = plot_idx // n_cols
            col = plot_idx  % n_cols

            ax_img = fig.add_subplot(n_rows, n_cols * 2, row * n_cols * 2 + col * 2 + 1)
            if self.dataset_handler.channels == 1:
                ax_img.imshow(
                    img.reshape(self.dataset_handler.rows, self.dataset_handler.cols),
                    cmap='gray'
                )
            else:
                ax_img.imshow(img)
            ax_img.set_title(f'True: {true_l}\nPredicted: {pred_l} ({conf:.2f})',
                             color='#e74c3c', fontsize=8, fontweight='bold')
            ax_img.axis('off')

            ax_bar = fig.add_subplot(n_rows, n_cols * 2, row * n_cols * 2 + col * 2 + 2)
            bar_colors = ['#e74c3c' if lbl == pred_l else ('#2ecc71' if lbl == true_l else '#bdc3c7') for lbl in labels]
            ax_bar.barh(labels, proba, color=bar_colors, height=0.6)
            ax_bar.set_xlim(0, 1.15)
            ax_bar.tick_params(axis='y', labelsize=6)
            ax_bar.tick_params(axis='x', labelsize=6)
            ax_bar.set_xlabel('P', fontsize=6)
            ax_bar.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            for i, (lbl, p) in enumerate(zip(labels, proba)):
                ax_bar.text(p + 0.02, i, f'{p:.2f}', va='center', fontsize=6,
                            color='#e74c3c' if lbl == pred_l else ('#2ecc71' if lbl == true_l else '#bdc3c7'),
                            fontweight='bold' if lbl == pred_l else 'normal')

        plt.suptitle(f'Top {top_n} Confident Misclassifications', y=1.01, fontsize=12, fontweight='bold')
        plt.tight_layout()
        self._save_fig('top_misclassified.png')

    def plot_emotion_confusion_heatmap(self, figsize: Tuple[int, int] = (10, 7)) -> None:
        """
        Off-diagonal confusion heatmap showing only misclassification rates.
        Diagonal is zeroed out to highlight confused emotion pairs (e.g. Fear vs Surprise).
        """
        if not self._guard(self.y_true is not None and self.y_pred is not None,
                       "No predictions available. Call predict() first."):
            return

        cm     = confusion_matrix(self.y_true, self.y_pred)
        cm_off = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        np.fill_diagonal(cm_off, 0)

        pairs = []
        for i in range(cm_off.shape[0]):
            for j in range(cm_off.shape[1]):
                if i != j and cm_off[i, j] > 0:
                    pairs.append((cm_off[i, j], i, j))
        pairs.sort(reverse=True)
        top3 = pairs[:3]

        fig, ax = plt.subplots(figsize=figsize)
        mask = cm_off == 0

        sns.heatmap(
            cm_off,
            annot=True, fmt='.1f', cmap='Reds',
            mask=mask,
            xticklabels=self.dataset_handler.class_labels,
            yticklabels=self.dataset_handler.class_labels,
            ax=ax,
            cbar_kws={'label': '% of misclassified'},
            square=True, linewidths=0.5, linecolor='whitesmoke',
        )
        ax.set_ylabel('True Class')

        plt.suptitle('Emotion Confusion Map')
        self._save_fig('emotion_confusion_heatmap.png')

    def plot_grad_cam_examples(self,
                               num_rows: int,
                               num_cols: int,
                               last_conv_layer_name: Optional[str],
                               mode: str,
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Grad-CAM visualization grid.
        mode='both'          – left half misclassified, right half correct
        mode='misclassified' – most confidently misclassified samples only
        mode='correct'       – most confidently correct samples only
        Each cell shows original image + Grad-CAM overlay side by side.
        """
        if not self._guard(self.y_true is not None and self.y_pred_proba is not None,
                           "No predictions available. Call predict() first."):
            return

        def _find_last_conv(model):
            all_layers = []
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    all_layers.extend(layer.layers)
                else:
                    all_layers.append(layer)
            for layer in reversed(all_layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer
            return None

        if last_conv_layer_name is None:
            conv_layer = _find_last_conv(self.model)
            if conv_layer is None:
                print("No Conv2D layer found in model.")
                return
            last_conv_layer_name = conv_layer.name
        else:
            conv_layer = None
            for layer in self.model.layers:
                if layer.name == last_conv_layer_name:
                    conv_layer = layer
                    break
                if hasattr(layer, 'layers'):
                    for sub in layer.layers:
                        if sub.name == last_conv_layer_name:
                            conv_layer = sub
                            break
                    if conv_layer is not None:
                        break
            if conv_layer is None:
                print(f"Layer '{last_conv_layer_name}' not found.")
                return

        print(f"Using last conv layer: {last_conv_layer_name}")

        top_n         = num_rows * num_cols
        misclassified = np.where(self.y_pred != self.y_true)[0]
        correct       = np.where(self.y_pred == self.y_true)[0]

        if mode == 'misclassified':
            if len(misclassified) == 0:
                print("No misclassifications found.")
                return
            conf           = np.max(self.y_pred_proba[misclassified], axis=1)
            selected       = misclassified[np.argsort(-conf)][:top_n]
            section_labels = [None]
            sections       = [selected]

        elif mode == 'correct':
            if len(correct) == 0:
                print("No correct predictions found.")
                return
            conf           = np.max(self.y_pred_proba[correct], axis=1)
            selected       = correct[np.argsort(-conf)][:top_n]
            section_labels = [None]
            sections       = [selected]

        elif mode == 'both':
            half = top_n // 2
            if len(misclassified) == 0:
                print("No misclassifications found, falling back to correct only.")
                mode     = 'correct'
                conf     = np.max(self.y_pred_proba[correct], axis=1)
                selected = correct[np.argsort(-conf)][:top_n]
                sections       = [selected]
                section_labels = [None]
            else:
                conf_m = np.max(self.y_pred_proba[misclassified], axis=1)
                conf_c = np.max(self.y_pred_proba[correct],       axis=1)
                sel_m  = misclassified[np.argsort(-conf_m)][:half]
                sel_c  = correct[np.argsort(-conf_c)][:half]
                sections       = [sel_m, sel_c]
                section_labels = ['Misclassified', 'Correct']
        else:
            print(f"Unknown mode '{mode}'. Choose 'both', 'misclassified', or 'correct'.")
            return

        all_indices = np.concatenate(sections)
        images      = self._load_images_by_indices(all_indices)

        def compute_gradcam(img_array: np.ndarray, class_idx: int) -> np.ndarray:
            inp = tf.cast(np.expand_dims(img_array, 0), tf.float32)

            captured      = {}
            original_call = conv_layer.call

            def hooked_call(*args, **kwargs):
                output = original_call(*args, **kwargs)
                captured['output'] = output
                return output

            conv_layer.call = hooked_call

            try:
                with tf.GradientTape() as tape:
                    tape.watch(inp)
                    predictions  = self.model(inp, training=False)
                    loss         = predictions[:, class_idx]
                    conv_outputs = captured['output']
            finally:
                conv_layer.call = original_call

            grads = tape.gradient(loss, conv_outputs)

            if grads is None:
                return np.zeros((img_array.shape[0], img_array.shape[1]))

            pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam    = conv_outputs[0] @ pooled[..., tf.newaxis]
            cam    = tf.squeeze(cam).numpy()
            cam    = np.maximum(cam, 0)

            if cam.max() > 0:
                cam /= cam.max()

            cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
            return cam

        figsize   = figsize or (num_cols * 4, num_rows * 3 + (0.5 if mode == 'both' else 0))
        fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=figsize)

        if num_rows == 1:
            axes = axes.reshape(1, -1)

        if mode == 'both' and len(sections) == 2:
            for label, x_fig in zip(section_labels, [0.02, 0.52]):
                fig.text(x_fig, 1.0, label,
                         fontsize=11, fontweight='bold',
                         color='#e74c3c' if label == 'Misclassified' else '#2ecc71',
                         va='top', ha='left',
                         transform=fig.transFigure)

            line = plt.Line2D([0.505, 0.505], [0.02, 0.98],
                              transform=fig.transFigure,
                              color='lightgray', linewidth=1.5, linestyle='--')
            fig.add_artist(line)

        flat_indices = []
        if mode == 'both' and len(sections) == 2:
            for row in range(num_rows):
                for col in range(num_cols // 2):
                    i = row * (num_cols // 2) + col
                    if i < len(sections[0]):
                        flat_indices.append(('mis', sections[0][i], row, col * 2))
                    if i < len(sections[1]):
                        flat_indices.append(('cor', sections[1][i], row, num_cols + col * 2))
        else:
            sel      = sections[0]
            tag_type = 'mis' if mode == 'misclassified' else 'cor'
            for idx in range(top_n):
                row = idx // num_cols
                col = idx  % num_cols
                if idx < len(sel):
                    flat_indices.append((tag_type, sel[idx], row, col * 2))

        for r in range(num_rows):
            for c in range(num_cols * 2):
                axes[r, c].axis('off')

        for entry in flat_indices:
            tag, sample_idx, row, base_col = entry
            img        = images[sample_idx]
            true_idx   = self.y_true[sample_idx]
            pred_idx   = self.y_pred[sample_idx]
            proba      = self.y_pred_proba[sample_idx]
            conf       = np.max(proba)
            true_label = self.dataset_handler.class_labels[true_idx]
            pred_label = self.dataset_handler.class_labels[pred_idx]

            cam = compute_gradcam(img, pred_idx)

            top_indices = np.argsort(proba)[::-1]
            alts        = [(self.dataset_handler.class_labels[i], proba[i])
                           for i in top_indices if i != pred_idx][:2]
            alt_str     = '  '.join([f'{l}: {p:.2f}' for l, p in alts])

            if tag == 'mis':
                color  = '#e74c3c' if conf > 0.9 else '#f39c12' if conf > 0.7 else '#e67e22'
                title  = f'True: {true_label}'
                xlabel = f'Pred: {pred_label} ({conf:.2f})'
            else:
                color  = '#2ecc71'
                title  = f'{true_label}'
                xlabel = f'Conf: {conf:.2f}'

            ax_orig = axes[row, base_col]
            ax_orig.imshow(img.squeeze(),
                           cmap='gray' if self.dataset_handler.channels == 1 else None)
            ax_orig.axis('off')
            ax_orig.set_title(title, fontsize=8, fontweight='bold')
            ax_orig.set_xlabel(xlabel, fontsize=7.5, color=color,
                               fontweight='bold', labelpad=3)

            ax_cam = axes[row, base_col + 1]
            if self.dataset_handler.channels == 1:
                base = np.stack([img.squeeze()] * 3, axis=-1)
            else:
                base = img

            heatmap = plt.cm.jet(cam)[..., :3]
            overlay = np.clip(0.55 * base + 0.45 * heatmap, 0, 1)

            ax_cam.imshow(overlay)
            ax_cam.axis('off')
            ax_cam.set_title('Grad-CAM', fontsize=8)
            ax_cam.set_xlabel(f'Alt: {alt_str}', fontsize=7, color='dimgray', labelpad=3)

        mode_str = {'both':          'Misclassified (left) vs Correct (right)',
                    'misclassified': 'Top Confident Misclassifications',
                    'correct':       'Top Confident Correct Predictions'}[mode]

        plt.suptitle(
            f'{mode_str}\n'
            f"Grad-CAM  (layer: {last_conv_layer_name})\n"
        )
        plt.tight_layout()
        self._save_fig(f'grad_cam_{mode}.png')

    def plot_per_epoch_class_f1(self, figsize: Tuple[int, int] = (14, 6)) -> None:
        """Line chart and heatmap of per-class F1-score over training epochs."""
        if not self._guard(self.epoch_class_f1 is not None,
                           "No epoch_class_f1 data. Pass it via __init__."):
            return

        epoch_class_f1 = self.epoch_class_f1
        epochs = range(1, len(epoch_class_f1) + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, self.dataset_handler.class_num))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        final_vals = {}
        for label, color in zip(self.dataset_handler.class_labels, colors):
            f1_vals = [d[label] for d in epoch_class_f1]
            ax1.plot(epochs, f1_vals, linewidth=2, color=color)
            final_vals[label] = (f1_vals[-1], color)

        sorted_vals = sorted(final_vals.items(), key=lambda x: x[1][0], reverse=True)

        text_areas = [
            TextArea(
                f'{label}: {val:.3f}',
                textprops=dict(color=color, fontsize=8,
                               family='monospace', fontweight='bold')
            )
            for label, (val, color) in sorted_vals
        ]
        packed = VPacker(children=text_areas, pad=2, sep=1)
        box = AnchoredOffsetbox(
            loc='upper left',
            child=packed,
            pad=0.5,
            frameon=True,
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax1.transAxes,
            borderpad=0,
        )
        box.patch.set(edgecolor='#333333', facecolor='white', alpha=0.85)
        ax1.add_artist(box)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('Per-Class F1 over Epochs')
        ax1.set_xlim(1, len(epochs))
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='both', alpha=0.3)

        matrix = np.array([
            [d[label] for label in self.dataset_handler.class_labels]
            for d in epoch_class_f1
        ]).T

        sns.heatmap(matrix, ax=ax2, cmap='YlGnBu',
                    xticklabels=[str(e) if e % 5 == 0 else '' for e in epochs],
                    yticklabels=self.dataset_handler.class_labels,
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'F1-Score'},
                    linewidths=0.3, linecolor='white')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Per-Class F1 Heatmap')

        plt.suptitle('Per-Class F1 over Training')
        self._save_fig('per_epoch_class_f1.png')

    def plot_top_misclassifications_chart(self, top_n: int,
                                          figsize: Tuple[int, int] = (10, 7)) -> None:
        """Horizontal bar chart of the most frequent true->predicted misclassification pairs."""
        top_pairs = self._get_top_misclassification_pairs(top_n)
        if not top_pairs:
            print("No predictions or misclassifications available.")
            return

        pair_labels  = [f"{p['true']} -> {p['predicted']}" for p in top_pairs]
        counts       = [p['count'] for p in top_pairs]
        pcts         = [p['pct_of_true'] for p in top_pairs]
        total_errors = int((self.y_pred != self.y_true).sum())
        total        = sum(counts)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos   = range(len(top_pairs))
        colors  = plt.cm.Reds(np.linspace(0.85, 0.3, len(top_pairs)))
        bars    = ax.barh(y_pos, counts, color=colors, alpha=0.85)

        for bar, count, pct in zip(bars, counts, pcts):
            ax.text(
                bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.1f}%)',
                va='center', fontsize=9
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Misclassifications')
        ax.set_title(
            f'Top {len(top_pairs)} Misclassification Pairs\n'
            f'Showing {total} / {total_errors} total errors ({total / total_errors * 100:.1f}%)',
            pad=20
        )
        ax.set_xlim(0, max(counts) * 1.25)
        ax.grid(axis='x', alpha=0.3)

        self._save_fig('misclassifications_chart.png')

    def plot_tsne_embeddings(self,
                             split: str,
                             layer_name: Optional[str],
                             n_samples: int,
                             perplexity: int) -> None:
        """
        t-SNE visualization of model's learned feature space.
        Works for both subclassed and functional/pretrained models.
        """
        generator_map = {
            'val':  self.data_augmentation_handler.val_generator,
            'test': self.data_augmentation_handler.test_generator,
        }
        generator = generator_map.get(split)
        if generator is None:
            raise ValueError(f"Invalid split '{split}'. Choose 'val' or 'test'.")

        def _find_layer(model, name):
            """Search top-level and one level of nested submodels."""
            try:
                return model.get_layer(name)
            except ValueError:
                pass
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        return layer.get_layer(name)
                    except ValueError:
                        pass
            return None

        if layer_name is None:
            preferred = [
                'global_average_pooling2d',   # SimpleCNN, MobileNetV2
                'avg_pool',                   # ResNet50
                'top_activation',             # EfficientNetB0
                'flatten',                    # VGG16
                'global_avg_pool',
            ]
            for name in preferred:
                if _find_layer(self.model, name) is not None:
                    layer_name = name
                    break

            if layer_name is None:
                for layer in reversed(self.model.layers[:-1]):
                    if isinstance(layer, (tf.keras.layers.Dense,
                                          tf.keras.layers.Dropout,
                                          tf.keras.layers.Softmax)):
                        continue
                    layer_name = layer.name
                    break

        if layer_name is None:
            print("Could not find a suitable feature layer.")
            return

        target_layer = _find_layer(self.model, layer_name)
        if target_layer is None:
            print(f"Layer '{layer_name}' not found in model or its submodels.")
            return

        print(f"Using layer: '{layer_name}'")

        def compute_features(img_batch: np.ndarray) -> np.ndarray:
            captured      = {}
            original_call = target_layer.call

            def hooked_call(*args, **kwargs):
                output = original_call(*args, **kwargs)
                captured['output'] = output
                return output

            target_layer.call = hooked_call
            try:
                self.model(img_batch, training=False)
            finally:
                target_layer.call = original_call

            if 'output' not in captured:
                raise RuntimeError(
                    f"Hook did not fire — layer '{layer_name}' may not "
                    f"be in the forward path for this input."
                )
            return captured['output'].numpy()

        generator.reset()
        features_list, labels_list, collected = [], [], 0

        for batch_x, batch_y in generator:
            feats = compute_features(batch_x)
            features_list.append(feats)
            labels_list.append(
                np.argmax(batch_y, axis=1) if batch_y.ndim > 1 else batch_y.astype(int)
            )
            collected += len(feats)
            if collected >= n_samples:
                break

        features = np.concatenate(features_list, axis=0)[:n_samples]
        labels   = np.concatenate(labels_list,   axis=0)[:n_samples]

        if features.ndim > 2:
            features = features.reshape(len(features), -1)

        print(f"Feature shape: {features.shape} | Samples: {len(labels)}")

        print("Running t-SNE (this may take a moment)...")
        embeddings = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=0,
        ).fit_transform(features)

        n_classes = self.dataset_handler.class_num
        colors    = plt.cm.tab10(np.linspace(0, 1, n_classes))

        fig, ax = plt.subplots(figsize=(10, 8))

        for class_idx, (label, color) in enumerate(
                zip(self.dataset_handler.class_labels, colors)):
            mask = labels == class_idx
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[color],
                label=label,
                alpha=0.6,
                s=15,
                linewidths=0,
            )

        for class_idx, label in enumerate(self.dataset_handler.class_labels):
            mask = labels == class_idx
            if mask.sum() > 0:
                cx = embeddings[mask, 0].mean()
                cy = embeddings[mask, 1].mean()
                ax.annotate(
                    label, (cx, cy),
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.7, edgecolor='gray')
                )

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(
            f"t-SNE Embeddings | {split.capitalize()}\n"
            f"({len(labels)} samples, layer: '{layer_name}', perplexity={perplexity})"
        )
        ax.legend(loc='upper right', markerscale=2)
        ax.grid(axis='both', alpha=0.3)

        self._save_fig(f'tsne_embeddings_{split}.png')

    # ── summary ──────────────────────────────────────────────

    def generate_summary(self, mode: str) -> None:
        if self.y_true is None or self.y_pred is None:
            print("No predictions available. Call predict() first.")
            return

        report    = self._get_classification_report()
        weighted  = report.get('weighted avg', {})
        macro     = report.get('macro avg', {})

        per_class_acc = self.per_class_acc
        best_class    = self.dataset_handler.class_labels[int(np.argmax(per_class_acc))]
        worst_class   = self.dataset_handler.class_labels[int(np.argmin(per_class_acc))]

        summary_data = [
            ('Model',                    CONFIG['model']),
            ('Strategy',                 CONFIG['strategy']),
            None,
            ('Test loss',                f"{self.test_loss:.4f}" if self.test_loss is not None else 'n/a'),
            ('Test accuracy',            f"{self.test_accuracy:.4f}" if self.test_accuracy is not None else 'n/a'),
            None,
            ('F1 weighted',              f"{weighted.get('f1-score', 0):.4f}"),
            ('F1 macro',                 f"{macro.get('f1-score', 0):.4f}"),
            ('Precision weighted',       f"{weighted.get('precision', 0):.4f}"),
            ('Recall weighted',          f"{weighted.get('recall', 0):.4f}"),
            ('Macro AUC',                f"{roc_auc_score(label_binarize(self.y_true, classes=range(self.dataset_handler.class_num)), self.y_pred_proba, average='macro'):.4f}"),
            None,
            ('Best class',               f"{best_class} ({per_class_acc.max():.4f})"),
            ('Worst class',              f"{worst_class} ({per_class_acc.min():.4f})"),
            ('Total misclassifications', int((self.y_pred != self.y_true).sum())),
            ('Misclassification rate',   f"{(self.y_pred != self.y_true).mean():.4f}"),
            ('Samples evaluated',        len(self.y_true)),
        ]

        if mode == 'latex':
            self._generate_latex_summary('EvaluationHandler', summary_data, 'evaluation_summary.tex')
        else:
            self._generate_ascii_summary('EvaluationHandler', summary_data)