import os
import random
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)

from .BaseHandler import BaseHandler
from .DatasetHandler import DatasetHandler


class DataAugmentationHandler(BaseHandler):
    """Manages data augmentation and generator creation for FER experiments."""

    def __init__(
            self,
            config: dict,
            dataset_handler: DatasetHandler,
            visualizations_directory: Optional[str] = None
        ):
        self.config          = config
        self.dataset_handler = dataset_handler
        super().__init__(visualizations_directory)

        self.train_datagen:  Optional[ImageDataGenerator] = None
        self.val_datagen:    Optional[ImageDataGenerator] = None
        self.test_datagen:   Optional[ImageDataGenerator] = None
        self.train_generator = None
        self.val_generator   = None
        self.test_generator  = None

        print('DataAugmentationHandler has been initialized.')

    # ── private helpers ─────────────────────────────────────

    @property
    def _aug_label(self) -> str:
        """Return augmentation preset name or 'No augmentation' if disabled."""
        aug = self.config['augmentation']
        if not aug['enabled']:
            return 'No augmentation'
        return aug.get('preset', 'Custom')

    def _build_train_datagen(self) -> ImageDataGenerator:
        """Build train ImageDataGenerator from current CONFIG augmentation settings."""
        aug = self.config['augmentation']

        if not aug['enabled']:
            return ImageDataGenerator(rescale=1./255)

        args = {
            'rescale':            1./255,
            'rotation_range':     aug['rotation_range'],
            'width_shift_range':  aug['width_shift_range'],
            'height_shift_range': aug['height_shift_range'],
            'zoom_range':         aug['zoom_range'],
            'horizontal_flip':    aug['horizontal_flip'],
            'fill_mode':          aug['fill_mode'],
        }

        if aug['brightness_range'][0] < aug['brightness_range'][1]:
            args['brightness_range'] = aug['brightness_range']

        if aug['shear_range'] > 0:
            args['shear_range'] = aug['shear_range']

        return ImageDataGenerator(**args)

    def _build_datagen_from_preset(self, preset: dict) -> ImageDataGenerator:
        """Build an ImageDataGenerator from an augmentation preset dict."""
        args = {
            'rescale':            1./255,
            'rotation_range':     preset.get('rotation_range', 0),
            'width_shift_range':  preset.get('width_shift_range', 0.0),
            'height_shift_range': preset.get('height_shift_range', 0.0),
            'zoom_range':         preset.get('zoom_range', 0.0),
            'horizontal_flip':    preset.get('horizontal_flip', False),
            'fill_mode':          preset.get('fill_mode', 'nearest'),
        }

        br = preset.get('brightness_range', [1.0, 1.0])
        if br[0] < br[1]:
            args['brightness_range'] = br

        shear = preset.get('shear_range', 0)
        if shear > 0:
            args['shear_range'] = shear

        return ImageDataGenerator(**args)

    def _color_mode(self) -> str:
        """Return Keras color mode string based on channel count."""
        return 'grayscale' if self.dataset_handler.channels == 1 else 'rgb'

    def _target_size(self) -> Tuple[int, int]:
        """Return (rows, cols) target size for image loading."""
        return (self.dataset_handler.rows, self.dataset_handler.cols)

    def _load_sample_image(self, class_name: Optional[str] = None, split: str = 'train'):
        """Load a random image from the given split and class."""
        if class_name is None:
            class_name = random.choice(self.dataset_handler.class_names)

        folder     = self.dataset_handler._folder(split)
        class_path = os.path.join(folder, class_name)
        images     = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        img_path   = os.path.join(class_path, random.choice(images))

        img       = load_img(img_path, target_size=self._target_size(), color_mode=self._color_mode())
        img_array = img_to_array(img).reshape((1,) + img_to_array(img).shape)
        return img, img_array, class_name

    def _show_image(self, ax, img_array: np.ndarray, title: str = '') -> None:
        """Display a single image on the given axes."""
        if self.dataset_handler.channels == 1:
            ax.imshow(img_array.reshape(self._target_size()), cmap='gray')
        else:
            ax.imshow(img_array)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # ── public ──────────────────────────────────────────────

    def create_generators(self) -> None:
        """
        Create train, val and test generators.
        """
        common = dict(
            target_size=self._target_size(),
            color_mode=self._color_mode(),
            class_mode='categorical',
            batch_size=self.config['batch_size'],
        )

        self.train_datagen   = self._build_train_datagen()
        self.train_generator = self.train_datagen.flow_from_directory(
            self.dataset_handler.train_folder, shuffle=True, **common
        )

        self.val_datagen   = ImageDataGenerator(rescale=1./255)
        self.val_generator = self.val_datagen.flow_from_directory(
            self.dataset_handler.val_folder, shuffle=False, **common
        )

        self.test_datagen   = ImageDataGenerator(rescale=1./255)
        self.test_generator = self.test_datagen.flow_from_directory(
            self.dataset_handler.test_folder, shuffle=False, **common
        )

        if hasattr(self.test_generator, 'index_array') and self.test_generator.index_array is not None:
            self._test_index_array_snapshot = self.test_generator.index_array.copy()
        else:
            self._test_index_array_snapshot = None

        print('Train, val and test generators have been created.')

    def reset_test_generator(self) -> None:
        """Restore test generator to its original post-creation state."""
        gen = self.test_generator

        if gen is None:
            return
        
        if self._test_index_array_snapshot is not None:
            gen.index_array = self._test_index_array_snapshot.copy()
            
        gen.index = 0

    # ── visualizations ──────────────────────────────────────

    def visualize_augmentation_with_maps(self, num_examples: int,
                                         figsize: Tuple[int, int] = (15, 8)) -> None:
        """Show augmented versions of one image alongside pixel difference maps."""
        if not self.config['augmentation']['enabled']:
            print("     Augmentation is disabled.")
            return

        img, img_array, class_name = self._load_sample_image()
        datagen  = self._build_train_datagen()
        original = img_to_array(img) / 255.0

        fig, axes = plt.subplots(2, num_examples + 1, figsize=figsize)

        self._show_image(axes[0, 0], original, f'Original\n({class_name.capitalize()})')
        axes[1, 0].axis('off')

        for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
            if i >= num_examples:
                break
            aug_img = batch[0]
            self._show_image(axes[0, i + 1], aug_img, f'Aug {i + 1}')

            diff         = np.abs(aug_img - original)
            diff_display = diff.reshape(self._target_size()) \
                if self.dataset_handler.channels == 1 else np.mean(diff, axis=2)
            im = axes[1, i + 1].imshow(diff_display, cmap='hot')
            axes[1, i + 1].set_title(f'Diff {i + 1}', fontsize=9)
            axes[1, i + 1].axis('off')
            plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

        plt.suptitle(f'Augmentation with Difference Maps | {self._aug_label} | {self.config["dataset"]}')
        self._save_fig('augmentation_with_maps.png')

    def visualize_augmentation_grid(self, num_images: int,
                                    augmentations_per_image: int,
                                    figsize: Tuple[int, int] = (15, 8)) -> None:
        """Grid of multiple source images with their augmented versions."""
        if not self.config['augmentation']['enabled']:
            print("     Augmentation is disabled.")
            return

        datagen   = self._build_train_datagen()
        fig, axes = plt.subplots(num_images, augmentations_per_image + 1, figsize=figsize)
        if num_images == 1:
            axes = axes.reshape(1, -1)

        for row in range(num_images):
            img, img_array, class_name = self._load_sample_image()

            axes[row, 0].imshow(img_to_array(img) / 255.0,
                                cmap='gray' if self.dataset_handler.channels == 1 else None)
            axes[row, 0].set_title(f'{class_name.capitalize()}\n(Original)', fontsize=10)
            axes[row, 0].axis('off')

            for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                if i >= augmentations_per_image:
                    break
                self._show_image(axes[row, i + 1], batch[0],
                                 f'Aug {i + 1}' if row == 0 else '')

        plt.suptitle(f'Augmentation Grid | {self._aug_label} | {self.config["dataset"]}')
        self._save_fig('augmentation_grid.png')

    def visualize_batch_examples(self, split: str, num_batches: int,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """Show real batches from the train, val or test generator."""
        generator_map = {
            'train': self.train_generator,
            'val':   self.val_generator,
            'test':  self.test_generator,
        }

        generator = generator_map.get(split)
        if generator is None:
            print("     Call create_generators() first.")
            return

        for batch_num in range(num_batches):
            images, labels = next(generator)
            batch_size     = min(len(images), 16)
            grid           = int(np.ceil(np.sqrt(batch_size)))

            fig, axes = plt.subplots(grid, grid, figsize=figsize)
            axes = axes.flatten()

            for i in range(batch_size):
                label = self.dataset_handler.class_labels[np.argmax(labels[i])]
                self._show_image(axes[i], images[i], label)

            for i in range(batch_size, len(axes)):
                axes[i].axis('off')

            plt.suptitle(
                f'{split.capitalize()} Batch {batch_num + 1} | {self._aug_label} | {self.config["dataset"]}'
            )
            self._save_fig(f'batch_examples_{split}_{batch_num + 1}.png')

    def visualize_augmentation_parameter_effects(self, num_examples: int,
                                                 figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Show the isolated effect of each augmentation parameter in a separate row.
        Column headers indicate the approximate transformation magnitude.
        """
        aug = CONFIG['augmentation']
        if not aug['enabled']:
            print("     Augmentation is disabled.")
            return

        img, img_array, class_name = self._load_sample_image()
        fill = aug['fill_mode']

        param_configs = []

        if aug['rotation_range'] > 0:
            r      = aug['rotation_range']
            angles = np.linspace(-r, r, num_examples)
            param_configs.append((
                f"Rotation +-{r} deg",
                {'rotation_range': r, 'fill_mode': fill},
                [f'rot ~{a:+.0f} deg' for a in angles],
            ))
        if aug['width_shift_range'] > 0 or aug['height_shift_range'] > 0:
            w      = aug['width_shift_range']
            h      = aug['height_shift_range']
            shifts = np.linspace(-max(w, h), max(w, h), num_examples)
            param_configs.append((
                f"Shift W:+-{w} H:+-{h}",
                {'width_shift_range': w, 'height_shift_range': h, 'fill_mode': fill},
                [f'shift ~{s:+.2f}' for s in shifts],
            ))
        if aug['zoom_range'] > 0:
            z     = aug['zoom_range']
            zooms = np.linspace(1 - z, 1 + z, num_examples)
            param_configs.append((
                f"Zoom +-{z}",
                {'zoom_range': z, 'fill_mode': fill},
                [f'zoom ~{zv:.2f}x' for zv in zooms],
            ))
        if aug.get('shear_range', 0) > 0:
            s      = aug['shear_range']
            shears = np.linspace(-s, s, num_examples)
            param_configs.append((
                f"Shear +-{s} deg",
                {'shear_range': s, 'fill_mode': fill},
                [f'shear ~{sv:+.2f}' for sv in shears],
            ))
        br = aug.get('brightness_range', [1.0, 1.0])
        if br[0] < br[1]:
            bvals = np.linspace(br[0], br[1], num_examples)
            param_configs.append((
                f"Brightness [{br[0]}, {br[1]}]",
                {'brightness_range': br},
                [f'bright ~{bv:.2f}' for bv in bvals],
            ))
        if aug.get('horizontal_flip', False):
            flip_titles = ['flipped' if i % 2 == 0 else 'original' for i in range(num_examples)]
            param_configs.append((
                'Horizontal Flip',
                {'horizontal_flip': True},
                flip_titles,
            ))

        if not param_configs:
            print("     No active augmentation parameters found in CONFIG.")
            return

        n_params  = len(param_configs)
        fig, axes = plt.subplots(n_params, num_examples + 1, figsize=figsize)
        if n_params == 1:
            axes = axes.reshape(1, -1)

        original = img_to_array(img) / 255.0

        for row, (param_label, aug_args, col_titles) in enumerate(param_configs):
            self._show_image(axes[row, 0], original, 'Original' if row == 0 else '')
            axes[row, 0].set_ylabel(param_label, fontsize=9, fontweight='bold',
                                    rotation=0, labelpad=120, va='center')

            datagen = ImageDataGenerator(rescale=1./255, **aug_args)
            for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                if i >= num_examples:
                    break
                title = col_titles[i] if row == 0 else ''
                self._show_image(axes[row, i + 1], batch[0], title)

        plt.suptitle(f'Isolated Augmentation Effects | {class_name.capitalize()} | {CONFIG["dataset"]}')
        self._save_fig('augmentation_parameter_effects.png')

    def visualize_augmentation_diversity(self, grid_size: int,
                                         figsize: Tuple[int, int] = (13, 13)) -> None:
        """Grid of grid_size x grid_size augmented versions of one image."""
        if not self.config['augmentation']['enabled']:
            print("     Augmentation is disabled.")
            return

        img, img_array, class_name = self._load_sample_image()
        datagen = self._build_train_datagen()
        n       = grid_size * grid_size

        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = axes.flatten()

        for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
            if i >= n:
                break
            self._show_image(axes[i], batch[0])
            axes[i].set_title(f'#{i+1}', fontsize=7, pad=2)

        plt.suptitle(
            f'Augmentation Diversity ({n} samples) | '
            f'{class_name.capitalize()} | {self._aug_label} | {self.config["dataset"]}'
        )
        self._save_fig('augmentation_diversity.png')

    def visualize_compare_strategies(self, strategies: List[Dict[str, Any]],
                                     strategy_names: List[str],
                                     num_examples: int,
                                     figsize: Tuple[int, int] = (16, 10)) -> None:
        """
        Compare multiple augmentation presets on the same image.
        Row 0 shows the original; each subsequent row shows one strategy.
        """
        if not strategies or len(strategies) != len(strategy_names):
            print("     strategies and strategy_names must be non-empty lists of equal length.")
            return

        img, img_array, class_name = self._load_sample_image()
        original = img_to_array(img) / 255.0

        n_rows = len(strategies) + 1
        n_cols = num_examples

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for j in range(n_cols):
            self._show_image(axes[0, j], original,
                             f'Original ({class_name.capitalize()})' if j == 0 else '')
        axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold',
                               rotation=0, labelpad=70, va='center')

        for i, (preset, name) in enumerate(zip(strategies, strategy_names), start=1):
            datagen = self._build_datagen_from_preset(preset)
            for j, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                if j >= num_examples:
                    break
                self._show_image(axes[i, j], batch[0], name if j == 0 else '')
            axes[i, 0].set_ylabel(name, fontsize=10, fontweight='bold',
                                   rotation=0, labelpad=70, va='center')

        plt.suptitle(f'Augmentation Strategy Comparison | {class_name.capitalize()} | {self.config["dataset"]}')
        self._save_fig('augmentation_strategy_comparison.png')

    def plot_pixel_intensity_comparison(self, num_samples: int,
                                        figsize: Tuple[int, int] = (14, 5)) -> None:
        """Histogram with KDE and CDF comparing pixel intensities before and after augmentation."""
        if not self.config['augmentation']['enabled']:
            print("     Augmentation is disabled.")
            return

        datagen = self._build_train_datagen()

        orig_pixels: List[np.ndarray] = []
        aug_pixels:  List[np.ndarray] = []

        collected = 0
        for class_name in self.dataset_handler.class_names:
            if collected >= num_samples:
                break
            try:
                img, img_array, _ = self._load_sample_image(class_name)
            except Exception:
                continue

            orig_pixels.append((img_to_array(img) / 255.0).flatten())
            batch = next(datagen.flow(img_array, batch_size=1))
            aug_pixels.append(batch[0].flatten())
            collected += 1

            while collected < num_samples:
                batch = next(datagen.flow(img_array, batch_size=1))
                aug_pixels.append(batch[0].flatten())
                orig_pixels.append((img_to_array(img) / 255.0).flatten())
                collected += 1
                break

        orig_all = np.concatenate(orig_pixels)
        aug_all  = np.concatenate(aug_pixels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bins = np.linspace(0, 1, 60)
        ax1.hist(orig_all, bins=bins, alpha=0.45, color='steelblue', density=True, label='Original')
        ax1.hist(aug_all,  bins=bins, alpha=0.45, color='orange',    density=True, label='Augmented')

        for data, color in [(orig_all, 'steelblue'), (aug_all, 'orange')]:
            kde = gaussian_kde(data, bw_method=0.05)
            xs  = np.linspace(0, 1, 300)
            ax1.plot(xs, kde(xs), color=color, linewidth=2)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel('Pixel Value (normalized)')
        ax1.set_ylabel('Density')
        ax1.set_title('Pixel Intensity Distribution')
        ax1.legend()
        ax1.grid(axis='both', alpha=0.3)

        for data, color, label in [
            (orig_all, 'steelblue', 'Original'),
            (aug_all,  'orange',    'Augmented'),
        ]:
            sorted_data = np.sort(data)
            cdf  = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            step = max(1, len(sorted_data) // 2000)
            ax2.plot(sorted_data[::step], cdf[::step], color=color, linewidth=2, label=label)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Pixel Value (normalized)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(axis='both', alpha=0.3)

        plt.suptitle(
            f'Pixel Intensity Comparison | {self._aug_label} | {self.config["dataset"]}\n'
            f'Sampling {num_samples} images'
        )
        self._save_fig('pixel_intensity_distribution.png')

    # ── summary ─────────────────────────────────────────────

    def generate_summary(self, mode: str) -> None:
        aug = self.config['augmentation']

        summary_data = [
            ('Augmentation enabled', str(aug['enabled'])),
            ('Augmentation preset',  aug.get('preset', 'Custom')),
            None,
            ('Rotation range',    f"+-{aug['rotation_range']} deg" if aug['enabled'] else 'n/a'),
            ('Width shift',       f"+-{aug['width_shift_range']*100:.0f}%" if aug['enabled'] else 'n/a'),
            ('Height shift',      f"+-{aug['height_shift_range']*100:.0f}%" if aug['enabled'] else 'n/a'),
            ('Zoom range',        f"{aug['zoom_range']*100:.0f}%" if aug['enabled'] else 'n/a'),
            ('Shear range',       f"{aug['shear_range']} deg" if aug['enabled'] else 'n/a'),
            ('Horizontal flip',   str(aug['horizontal_flip']) if aug['enabled'] else 'n/a'),
            ('Brightness range',  str(aug['brightness_range']) if aug['enabled'] else 'n/a'),
            ('Fill mode',         aug['fill_mode'] if aug['enabled'] else 'n/a'),
            None,
            ('Batch size',  self.config['batch_size']),
            ('Color mode',  self._color_mode()),
            ('Target size', f'{self._target_size()[0]} x {self._target_size()[1]}'),
        ]

        if mode == 'latex':
            self._generate_latex_summary('DataAugmentationHandler', summary_data, 'data_augmentation_summary.tex')
        else:
            self._generate_ascii_summary('DataAugmentationHandler', summary_data)