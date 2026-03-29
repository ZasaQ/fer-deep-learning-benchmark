import os
import random
import zipfile
from typing import Optional, List, Dict, Tuple, Any

import cv2
import gdown
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import shutil

from .BaseHandler import BaseHandler


class DatasetHandler(BaseHandler):
    """Manages dataset structure, metadata and visualizations for FER experiments."""

    SPLITS = ('train', 'val', 'test')
    DATASET_LINKS = {
        'FER2013'   : 'https://drive.google.com/file/d/1EXbZCq_9xBr57AMK131DxrEWLMDjIGfV/view?usp=sharing',
        'CK+'       : 'https://drive.google.com/file/d/1_3fCNlRjIzgjEPMjbMVeDJFozBzB9d7U/view?usp=sharing',
        'RAF-DB'    : 'https://drive.google.com/file/d/1GhY4jSNhznDTyJ9zNd0SltELswta0SM9/view?usp=sharing',
        'AffectNet' : 'https://drive.google.com/file/d/1rD3ZkaLdjrwBHsq3gLUXUTKRRoSRGx43/view?usp=sharing'
    }

    def __init__(self, config: dict, visualizations_directory: Optional[str] = None):
        super().__init__(visualizations_directory)
        self.config = config

        self.dataset_name = ''

        self.train_folder = 'dataset/train'
        self.val_folder   = 'dataset/val'
        self.test_folder  = 'dataset/test'

        self.resolution_info: dict = {}
        self._channels: int = 3

        self.class_names: List[str] = []
        self.class_labels: List[str] = []
        self.class_num: int = 0
        self.rows: int = 0
        self.cols: int = 0

        print('DatasetHandler has been initialized.')

    # ── private ─────────────────────────────────────────────

    def _discover_classes(self) -> None:
        """Read class names from train folder subdirectories."""
        if not os.path.exists(self.train_folder):
            raise FileNotFoundError(f"Train folder not found: {self.train_folder}")

        self.class_names = sorted([
            d for d in os.listdir(self.train_folder)
            if os.path.isdir(os.path.join(self.train_folder, d))
        ])

        if not self.class_names:
            raise ValueError(f"No class directories found in: {self.train_folder}")

        self.class_num    = len(self.class_names)
        self.class_labels = [name.capitalize() for name in self.class_names]

    def _discover_image_dimensions(self) -> None:
        """Read image dimensions from the train folder."""
        size_counts = {}
        for class_name in self.class_names:
            class_path = os.path.join(self.train_folder, class_name)
            for f in os.listdir(class_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with Image.open(os.path.join(class_path, f)) as img:
                        size_counts[img.size] = size_counts.get(img.size, 0) + 1

        if not size_counts:
            raise ValueError(f"No images found in: {self.train_folder}")

        total    = sum(size_counts.values())
        dominant = max(size_counts, key=size_counts.get)
        dominant_pct = size_counts[dominant] / total * 100

        self.cols, self.rows = dominant

        self.resolution_info = {
            'dominant':     f'{dominant[0]} x {dominant[1]}',
            'dominant_pct': dominant_pct,
            'unique_count': len(size_counts),
            'total_images': total,
        }

        if len(size_counts) == 1:
            self.resolution_info['summary'] = f'{dominant[0]} x {dominant[1]} (uniform)'
        else:
            sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[1])
            top3  = sorted_sizes[:3]
            parts = [f'{w}x{h}: {c} ({c/total*100:.1f}%)' for (w, h), c in top3]
            remaining = len(size_counts) - 3
            suffix = f' +{remaining} more' if remaining > 0 else ''
            self.resolution_info['summary'] = f'mixed ({", ".join(parts)}{suffix})'

    def _discover_color_mode(self) -> None:
        mode_counts: dict = {}

        for class_name in self.class_names:
            class_path = os.path.join(self.train_folder, class_name)
            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            for img_name in images[:20]:
                with Image.open(os.path.join(class_path, img_name)) as img:
                    mode_counts[img.mode] = mode_counts.get(img.mode, 0) + 1

        grayscale = mode_counts.get('L', 0)
        color     = sum(v for k, v in mode_counts.items() if k != 'L')

        self._channels = 1 if grayscale > color else 3

        if self._channels == 1 and self.config.get('dataset') == 'FER2013' and self.config.get('model') != 'SimpleCNN':
            self._channels = 3

    def _discover_dominant_format(self) -> str:
        """Check image formats across train folder and return dominant format string."""
        ext_counts = {}
        for class_name in self.class_names:
            class_path = os.path.join(self.train_folder, class_name)
            for f in os.listdir(class_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'):
                    normalized = 'JPEG' if ext in ('.jpg', '.jpeg') else ext[1:].upper()
                    ext_counts[normalized] = ext_counts.get(normalized, 0) + 1

        if not ext_counts:
            return 'unknown'

        total        = sum(ext_counts.values())
        dominant     = max(ext_counts, key=ext_counts.get)
        dominant_pct = ext_counts[dominant] / total * 100

        if dominant_pct == 100:
            return dominant
        elif dominant_pct > 80:
            return f'{dominant} (mostly, {dominant_pct:.0f}%)'
        else:
            parts = [f'{fmt}: {count}' for fmt, count in sorted(ext_counts.items(), key=lambda x: -x[1])]
            return f'mixed ({", ".join(parts)})'

    def _folder(self, split: str) -> str:
        """Return folder path for a given split name."""
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got: '{split}'")
        return {'train': self.train_folder,
                'val':   self.val_folder,
                'test':  self.test_folder}[split]

    # ── properties ──────────────────────────────────────────

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.rows, self.cols, self._channels)

    # ── public ──────────────────────────────────────────────

    def download_dataset(self) -> None:
        self.dataset_name = self.config['dataset']
        dest = f'/content/datasets/{self.dataset_name}'

        if not os.path.isdir(dest):
            raise FileNotFoundError(
                f"Dataset '{self.dataset_name}' not found at {dest}.\n"
                "Run the dataset download cell first."
            )

        if os.path.islink('dataset'):
            os.unlink('dataset')
        elif os.path.isdir('dataset'):
            shutil.rmtree('dataset')

        os.symlink(dest, 'dataset')
        print(f"Dataset switched to: {self.dataset_name} → {dest}")


    def discover_dataset(self) -> None:
        """Discover class names, image dimensions and color mode from the dataset folder."""
        self._discover_classes()
        self._discover_image_dimensions()
        self._discover_color_mode()

        print(f'Classes ({self.class_num}): {self.class_labels}')
        print(f'Image size : {self.cols} x {self.rows}')
        print(f'Channels   : {self.channels}  ({"grayscale" if self.channels == 1 else "RGB"})')
        print(f'Input shape: {self.input_shape}')

    def get_class_distribution(self, split: str) -> Dict[str, int]:
        """Return image count per class for a given split."""
        folder = self._folder(split)
        return {
            class_name: len([
                f for f in os.listdir(os.path.join(folder, class_name))
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            for class_name in self.class_names
        }

    def validate_structure(self) -> bool:
        """Check that all splits exist and contain the same class folders."""
        for split in self.SPLITS:
            folder = self._folder(split)

            if not os.path.exists(folder):
                print(f"  {split} folder not found: {folder}")
                return False

            classes = sorted([
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ])

            if classes != self.class_names:
                print(f"  Class mismatch in '{split}': expected {self.class_names}, got {classes}")
                return False

        print("Dataset structure valid\n")
        for split in self.SPLITS:
            print(f"{split.capitalize()} distribution:")
            for name, count in self.get_class_distribution(split).items():
                print(f"  {name}: {count}")
            print()

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'train_folder': self.train_folder,
            'val_folder':   self.val_folder,
            'test_folder':  self.test_folder,
            'class_names':  self.class_names,
            'class_labels': self.class_labels,
            'class_num':    self.class_num,
            'image_size':   (self.cols, self.rows),
            'channels':     self.channels,
            'input_shape':  self.input_shape,
        }

    # ── visualizations ──────────────────────────────────────

    def plot_class_distribution(self, split: Optional[str], figsize: Tuple[int, int] = (8, 5)) -> None:
        """Bar chart of image count per class. If split is None, shows all splits in separate panels."""
        splits_to_plot = [split] if split else self.SPLITS
        n_splits = len(splits_to_plot)
        colors   = {'train': 'steelblue', 'val': 'mediumseagreen', 'test': 'coral'}

        fig, axes = plt.subplots(
            1, n_splits,
            figsize=(figsize[0] * n_splits if n_splits > 1 else figsize[0], figsize[1]),
            sharey=(n_splits > 1)
        )

        if n_splits == 1:
            axes = [axes]

        all_values = [v for s in splits_to_plot for v in self.get_class_distribution(s).values()]
        max_val    = max(all_values)

        for ax, s in zip(axes, splits_to_plot):
            color = colors.get(s, 'steelblue')
            dist  = self.get_class_distribution(s)
            total = sum(dist.values())
            bars  = ax.bar(range(len(dist)), list(dist.values()), color=color, alpha=0.8)
            ax.set_xticks(range(len(dist)))
            ax.set_xticklabels(self.class_labels, rotation=45, ha='right')
            ax.set_ylabel('Number of Images')
            ax.set_ylim(0, max_val * 1.25)
            ax.set_title(f'{s.capitalize()} Set (n={total:,})' if not split else f'n={total:,}')
            ax.grid(axis='y', alpha=0.3)

            for bar, count in zip(bars, dist.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{count}\n({count/total*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)

        title_split = split.capitalize() if split else "All Splits"
        plt.suptitle(f"Class Distribution | {title_split} | {self.config['dataset']}")
        plt.tight_layout()

        filename = f'class_distribution_{split}.png' if split else 'class_distribution.png'
        self._save_fig(filename)

    def plot_class_distribution_comparison(self, figsize: Tuple[int, int] = (13, 6)) -> None:
        """Grouped bar chart comparing class counts across all three splits."""
        dists   = {s: self.get_class_distribution(s) for s in self.SPLITS}
        x       = np.arange(len(self.class_labels))
        width   = 0.25
        colors  = ['steelblue', 'mediumseagreen', 'coral']
        offsets = [-width, 0, width]

        fig, ax     = plt.subplots(figsize=figsize)
        grand_total = 0

        for split, color, offset in zip(self.SPLITS, colors, offsets):
            total       = sum(dists[split].values())
            grand_total += total
            bars = ax.bar(x + offset, list(dists[split].values()), width,
                          label=f'{split.capitalize()} (n={total:,})',
                          color=color, alpha=0.8)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{int(bar.get_height())}',
                        ha='center', va='bottom', fontsize=7)

        ax.set_ylim(0, max(max(dists[s].values()) for s in self.SPLITS) * 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax.set_ylabel('Number of Images')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.suptitle(f"Combined Class Distribution | {self.config['dataset']}\nTotal Images Amount: {grand_total}")
        self._save_fig('split_distribution_comparison.png')

    def plot_class_distribution_pie(self, split: str, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Pie chart of class distribution for a given split."""
        distribution = self.get_class_distribution(split)
        total   = sum(distribution.values())
        colors  = plt.cm.tab10(np.linspace(0, 1, len(distribution)))
        explode = [0.05] * len(distribution)

        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            distribution.values(),
            labels=self.class_labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=explode,
        )
        plt.suptitle(f'{split.capitalize()} Set Distribution | {self.config["dataset"]}')
        ax.legend(
            [f'{l}: {c} ({c/total*100:.1f}%)' for l, c in zip(self.class_labels, distribution.values())],
            loc='center left', bbox_to_anchor=(1, 0.5)
        )
        self._save_fig(f'class_distribution_pie_{split}.png')

    def plot_sample_images(self, samples_per_class: int, split: str,
                           figsize: Optional[Tuple[int, int]] = None) -> None:
        """Grid of random sample images, one row per class."""
        folder  = self._folder(split)
        figsize = figsize or (samples_per_class * 2, self.class_num * 2)

        fig, axes = plt.subplots(self.class_num, samples_per_class, figsize=figsize)

        if self.class_num == 1:
            axes = axes.reshape(1, -1)

        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(folder, class_name)
            images  = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            samples = random.sample(images, min(samples_per_class, len(images)))

            for j in range(samples_per_class):
                ax = axes[i, j]
                if j < len(samples):
                    img = Image.open(os.path.join(class_path, samples[j]))
                    ax.imshow(img, cmap='gray' if self.channels == 1 else None)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if j == 0:
                    ax.set_ylabel(self.class_labels[i], rotation=0, labelpad=40,
                                  fontsize=12, fontweight='bold')

        plt.suptitle(f'Sample Images | {split.capitalize()} | {self.config["dataset"]}', y=1.01)
        self._save_fig('image_samples.png')

    def plot_split_ratio(self, figsize: Tuple[int, int] = (13, 5)) -> None:
        """Stacked percentage bars per class and overall split pie chart."""
        dists  = {s: self.get_class_distribution(s) for s in self.SPLITS}
        totals = {s: sum(dists[s].values()) for s in self.SPLITS}
        colors = ['steelblue', 'mediumseagreen', 'coral']

        grand = {cn: sum(dists[s][cn] for s in self.SPLITS) for cn in self.class_names}
        pcts  = {
            s: [dists[s][cn] / grand[cn] * 100 for cn in self.class_names]
            for s in self.SPLITS
        }

        x = np.arange(len(self.class_labels))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bottoms = np.zeros(len(self.class_names))
        for split, color in zip(self.SPLITS, colors):
            ax1.bar(x, pcts[split], bottom=bottoms,
                    label=split.capitalize(), color=color, alpha=0.8)
            for i in range(len(self.class_names)):
                ax1.text(x[i], bottoms[i] + pcts[split][i] / 2,
                         f'{pcts[split][i]:.1f}%', ha='center', va='center', fontsize=7)
            bottoms += np.array(pcts[split])

        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_ylim(0, 105)
        ax1.set_title('Train / Val / Test Split by Class')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        ax2.pie(
            [totals[s] for s in self.SPLITS],
            labels=[f'{s.capitalize()}\n(n={totals[s]:,})' for s in self.SPLITS],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
        )
        ax2.set_title('Overall Split')

        plt.suptitle(f"Split Ratio | {self.config['dataset']}")
        self._save_fig('split_ratio.png')

    def plot_class_imbalance_analysis(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Normalised class distribution with max/min imbalance ratio per split."""
        dists   = {s: self.get_class_distribution(s) for s in self.SPLITS}
        arrs    = {s: np.array(list(dists[s].values())) for s in self.SPLITS}
        colors  = ['steelblue', 'mediumseagreen', 'coral']
        x       = np.arange(len(self.class_labels))
        width   = 0.25
        offsets = [-width, 0, width]

        fig, ax = plt.subplots(figsize=figsize)

        balanced = 100 / self.class_num
        for split, color, offset in zip(self.SPLITS, colors, offsets):
            ax.bar(x + offset, arrs[split] / arrs[split].sum() * 100, width,
                   label=split.capitalize(), color=color, alpha=0.8)

        ax.axhline(balanced, color='black', linestyle='--', alpha=0.5)
        ax.text(len(x) - 0.5, balanced + 0.2,
                f'Balanced: {balanced:.1f}%', color='black', fontweight='bold',
                fontsize=8, ha='right', va='bottom')

        ratio_lines = '\n'.join(
            f'{s.capitalize()} ratio: {arrs[s].max() / arrs[s].min():.2f}×'
            for s in self.SPLITS
        )
        ax.text(0.02, 0.97, ratio_lines,
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='lightyellow', edgecolor='gray'))

        ax.set_xticks(x)
        ax.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, max(np.max(arrs[s] / arrs[s].sum() * 100) for s in self.SPLITS) * 1.15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f"Imbalance Analysis | {self.config['dataset']}")
        self._save_fig('class_imbalance_analysis.png')

    def plot_pixel_intensity_distribution(self, split: str, num_samples: int,
                                          figsize: Tuple[int, int] = (14, 5)) -> None:
        """Overall histogram, per-class KDE overlay of pixel intensities."""
        folder            = self._folder(split)
        samples_per_class = num_samples // self.class_num
        all_pixels        = []
        class_pixels      = {cn: [] for cn in self.class_names}

        print(f"Sampling {num_samples} images from '{split}' for pixel analysis...")

        for class_name in self.class_names:
            class_path = os.path.join(folder, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in random.sample(images, min(samples_per_class, len(images))):
                img    = Image.open(os.path.join(class_path, img_name))
                img    = img.convert('L' if self.channels == 1 else 'RGB')
                pixels = np.array(img).flatten()
                all_pixels.extend(pixels)
                class_pixels[class_name].extend(pixels)

        all_pixels_arr = np.array(all_pixels)
        colors         = plt.cm.tab10(np.linspace(0, 1, self.class_num))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.hist(all_pixels_arr, bins=50, color='steelblue', alpha=0.75, edgecolor='none')
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Pixel Intensity', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title('Overall Pixel Distribution', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.text(0.97, 0.97,
                f'mean: {all_pixels_arr.mean():.2f}\nstd:  {all_pixels_arr.std():.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

        ax = axes[1]
        x_grid = np.linspace(0, 255, 512)
        for class_name, color in zip(self.class_names, colors):
            px  = np.array(class_pixels[class_name], dtype=np.float32)
            kde = gaussian_kde(px, bw_method=0.15)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.25, color=color)
            ax.plot(x_grid, kde(x_grid), lw=1.4, color=color,
                    label=class_name.capitalize())
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Pixel Intensity', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title('Per-Class Pixel Distributions', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, framealpha=0.85, loc='upper left',
                  handlelength=1.2, labelspacing=0.3)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

        plt.suptitle(
            f'Pixel Intensity Distribution | {split.capitalize()} | {self.config["dataset"]}\n'
            f'Sampling {num_samples} images'
        )
        self._save_fig('pixel_intensity_distribution.png')

    def plot_class_heatmap(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """Heatmap of image counts and percentages per class for all three splits."""
        dists  = {s: self.get_class_distribution(s) for s in self.SPLITS}
        data   = np.array([list(dists[s].values()) for s in self.SPLITS])
        totals = {s: sum(dists[s].values()) for s in self.SPLITS}

        annot_data = np.array([
            [f'{v}\n({v/totals[s]*100:.1f}%)' for v in dists[s].values()]
            for s in self.SPLITS
        ])

        total_col  = np.array([[totals[s]] for s in self.SPLITS])
        data_full  = np.hstack([data, total_col])
        annot_full = np.hstack([
            annot_data,
            [[f'{totals[s]}\n(100%)'] for s in self.SPLITS]
        ])

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data_full,
            annot=annot_full,
            fmt='',
            cmap='YlOrRd',
            xticklabels=self.class_labels + ['Total'],
            yticklabels=[s.capitalize() for s in self.SPLITS],
            ax=ax,
            cbar_kws={'label': 'Image Count'},
            linewidths=0.5,
            linecolor='white',
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f"Class Balance Heatmap | {self.config['dataset']}")
        self._save_fig('class_heatmap.png')

    def plot_tsne_embeddings(self,
                         split: str,
                         num_samples: int,
                         perplexity: int,
                         figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        2D embedding of raw pixel data using t-SNE.
        Shows class separability in raw feature space before any model training.
        """
        folder            = self._folder(split)
        samples_per_class = num_samples // self.class_num
        pixels_list       = []
        labels_list       = []

        print(f"Loading {num_samples} images from '{split}' for t-SNE embedding...")

        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(folder, class_name)
            images  = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            sampled = random.sample(images, min(samples_per_class, len(images)))

            for img_name in sampled:
                img = Image.open(os.path.join(class_path, img_name))
                img = img.convert('L' if self.channels == 1 else 'RGB')
                img = img.resize((self.cols, self.rows))
                pixels_list.append(np.array(img).flatten())
                labels_list.append(class_idx)

        X = np.array(pixels_list, dtype=np.float32) / 255.0
        y = np.array(labels_list)

        print(f"Running t-SNE on {X.shape[0]} samples ({X.shape[1]} features each)...")

        embedding = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(X) - 1),
            random_state=42,
            max_iter=1000,
            init='pca',
            learning_rate='auto',
        ).fit_transform(X)

        fig, ax = plt.subplots(figsize=figsize)
        cmap    = plt.colormaps['tab10'].resampled(self.class_num)

        for class_idx, class_label in enumerate(self.class_labels):
            mask = y == class_idx
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap(class_idx)],
                label=class_label,
                alpha=0.55, s=18, edgecolors='none',
            )
            cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
            ax.text(cx, cy, class_label, fontsize=8, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2',
                            facecolor='white', alpha=0.7, edgecolor='gray'))

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(
            f't-SNE Embedding of Raw Pixels | '
            f'{split.capitalize()} | {self.config["dataset"]}\n'
            f'({X.shape[0]} samples, {X.shape[1]}D → 2D, perplexity={perplexity})'
        )
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                fontsize=10, markerscale=2.5, framealpha=0.9)
        ax.grid(axis='both', alpha=0.3)
        self._save_fig('tsne_embedding.png')
        print("t-SNE embedding complete.")

    def plot_image_quality_analysis(self, split: str, num_samples: int,
                                    figsize: Tuple[int, int] = (14, 5)) -> None:
        """Per-class sharpness analysis using Laplacian variance."""
        folder            = self._folder(split)
        samples_per_class = max(1, num_samples // self.class_num)
        class_sharpness   = {cn: [] for cn in self.class_names}

        print(f"Analyzing image sharpness ({num_samples} samples from '{split}')...")

        for class_name in self.class_names:
            class_path = os.path.join(folder, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in random.sample(images, min(samples_per_class, len(images))):
                img = Image.open(os.path.join(class_path, img_name)).convert('L')
                arr = np.array(img, dtype=np.float64)
                laplacian_var = cv2.Laplacian(arr.astype(np.uint8), cv2.CV_64F).var()
                class_sharpness[class_name].append(laplacian_var)

        colors = plt.cm.tab10(np.linspace(0, 1, self.class_num))
        data   = [class_sharpness[cn] for cn in self.class_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bp = ax1.boxplot(data, positions=range(self.class_num),
                         patch_artist=True,
                         medianprops=dict(color='black'),
                         flierprops=dict(marker='o', markersize=3, alpha=0.3))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        all_vals     = [v for vals in class_sharpness.values() for v in vals]
        overall_mean = np.mean(all_vals)
        upper_limit  = np.percentile(all_vals, 95) * 1.2
        ax1.set_ylim(0, upper_limit)
        mean_line = ax1.axhline(overall_mean, color='black', linestyle='--', alpha=0.5)
        ax1.annotate(
            f'Overall mean: {overall_mean:.1f}',
            xy=(self.class_num - 1, overall_mean),
            xytext=(-8, 8), textcoords='offset points',
            fontsize=8, fontweight='bold', color='black',
            ha='right', va='bottom',
        )
        ax1.legend(handles=[mean_line], loc='upper right', fontsize=8,
                   framealpha=0.85, edgecolor='gray')
        ax1.set_xticks(range(self.class_num))
        ax1.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax1.set_ylabel('Laplacian Variance')
        ax1.set_title('Sharpness Box Plot by Class')
        ax1.grid(axis='y', alpha=0.3)

        parts = ax2.violinplot(data, positions=range(self.class_num),
                               showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        for key in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
            if key in parts:
                parts[key].set_color('black')

        ax2_upper = np.percentile(all_vals, 100) * 1.2
        ax2.set_ylim(0, ax2_upper)
        for i, (cn, color) in enumerate(zip(self.class_names, colors)):
            vals = class_sharpness[cn]
            cv   = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0
            ax2.annotate(
                f'CV={cv:.2f}',
                xy=(i, ax2_upper * 0.97),
                xytext=(0, 0), textcoords='offset points',
                fontsize=7, ha='center', va='top', color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.80),
            )

        ax2.set_xticks(range(self.class_num))
        ax2.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax2.set_ylabel('Laplacian Variance')
        ax2.set_title('Sharpness Distribution by Class')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(f"Image Quality Analysis | {split.capitalize()} | {self.config['dataset']}\nSampling {num_samples} images")
        self._save_fig('image_quality_analysis.png')

    def plot_face_brightness_analysis(self, split: str, num_samples: int,
                                      figsize: Tuple[int, int] = (14, 5)) -> None:
        """Per-class mean brightness (average pixel value) with KDE curves."""
        folder            = self._folder(split)
        samples_per_class = max(1, num_samples // self.class_num)
        class_brightness  = {cn: [] for cn in self.class_names}

        print(f"Analyzing face brightness ({num_samples} samples from '{split}')...")

        for class_name in self.class_names:
            class_path = os.path.join(folder, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in random.sample(images, min(samples_per_class, len(images))):
                img = Image.open(os.path.join(class_path, img_name)).convert('L')
                class_brightness[class_name].append(np.mean(np.array(img)))

        means  = [np.mean(class_brightness[cn]) for cn in self.class_names]
        colors = plt.cm.tab10(np.linspace(0, 1, self.class_num))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bars = ax1.bar(range(self.class_num), means, color=colors, alpha=0.8)
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{mean:.1f}', ha='center', va='bottom', fontsize=8)

        overall_mean = np.mean([v for vals in class_brightness.values() for v in vals])
        ax1.axhline(overall_mean, color='black', linestyle='--', alpha=0.7)
        ax1.text(self.class_num - 0.5, overall_mean + 1,
                 f'Overall: {overall_mean:.1f}', color='black', fontweight='bold',
                 fontsize=8, ha='right')
        ax1.set_xticks(range(self.class_num))
        ax1.set_xticklabels(self.class_labels, rotation=45, ha='right')
        ax1.set_ylabel('Mean Pixel Value (0–255)')
        ax1.set_title('Mean Face Brightness by Class')
        ax1.set_ylim(0, 260)
        ax1.grid(axis='y', alpha=0.3)

        for class_name, color in zip(self.class_names, colors):
            vals = np.array(class_brightness[class_name])
            kde  = gaussian_kde(vals, bw_method=0.3)
            x    = np.linspace(0, 255, 300)
            ax2.plot(x, kde(x), linewidth=2, label=class_name.capitalize(), color=color)
            ax2.fill_between(x, kde(x), alpha=0.1, color=color)

        ax2.set_xlim(0, 255)
        ax2.set_ylim(bottom=0)
        ax2.set_xlabel('Mean Pixel Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Brightness Density per Class')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(
            f'Face Brightness Analysis | {split.capitalize()} | {self.config["dataset"]}\n'
            f'Sampling {num_samples} images'
        )
        self._save_fig('face_brightness_analysis.png')

    # ── summary ─────────────────────────────────────────────

    def generate_summary(self, mode: str) -> None:
        """Generate dataset summary."""
        dists  = {s: self.get_class_distribution(s) for s in self.SPLITS}
        totals = {s: sum(dists[s].values()) for s in self.SPLITS}
        grand  = sum(totals.values())
        arrs   = {s: np.array(list(dists[s].values())) for s in self.SPLITS}

        summary_data = [
            ('Dataset',                  self.config['dataset']),
            None,
            ('Classes',                  self.class_num),
            ('Class labels',             ', '.join(self.class_labels)),
            None,
            ('Input shape',              f'{self.rows} x {self.cols} x {self.channels}'),
            ('Color mode',               'grayscale' if self.channels == 1 else 'rgb'),
            ('Image format',             self._discover_dominant_format()),
            ('Resolution',               self.resolution_info.get('summary', f'{self.cols} x {self.rows}')),
            ('Unique resolutions',       self.resolution_info.get('unique_count', 1)),
            None,
            ('Imbalance ratio (train)',  f'{arrs["train"].max()/arrs["train"].min():.2f} : 1'),
            ('Imbalance ratio (val)',    f'{arrs["val"].max()/arrs["val"].min():.2f} : 1'),
            ('Imbalance ratio (test)',   f'{arrs["test"].max()/arrs["test"].min():.2f} : 1'),
            None,
            ('Train images',             f'{totals["train"]:,} ({totals["train"]/grand*100:.1f}%)'),
            ('Val images',               f'{totals["val"]:,} ({totals["val"]/grand*100:.1f}%)'),
            ('Test images',              f'{totals["test"]:,} ({totals["test"]/grand*100:.1f}%)'),
            ('Total',                    f'{grand:,}'),
        ]

        if mode == 'latex':
            self._generate_latex_summary('DatasetHandler', summary_data, 'dataset_summary.tex')
        else:
            self._generate_ascii_summary('DatasetHandler', summary_data)