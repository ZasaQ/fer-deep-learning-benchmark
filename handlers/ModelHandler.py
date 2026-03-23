from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from .BaseHandler import BaseHandler
from .DatasetHandler import DatasetHandler
from ExperimentMetrics import ModelMetricsMixin


class ModelHandler(ModelMetricsMixin, BaseHandler):
    """
    Builds Keras models for all five FER architectures.

    Models:     SimpleCNN, VGG16, ResNet50, MobileNetV2, EfficientNetB0
    Strategies: baseline (SimpleCNN only), tl, pft, fft
    """

    # Layer index at which PFT unfreezing begins (architecture-specific)
    PFT_LAYER_INDEX = {
        'resnet50':       140,
        'mobilenetv2':    100,
        'efficientnetb0': 200,
    }

    # VGG16 uses named-block unfreezing instead of index-based
    VGG16_PFT_BLOCKS = ('block4', 'block5')

    TRANSFER_STRATEGIES = ('tl', 'pft', 'fft')

    def __init__(self, config: dict, dataset_handler: DatasetHandler):
        self.config          = config
        self.dataset_handler = dataset_handler
        self.model: Optional[tf.keras.Model] = None

        super().__init__()
        print('ModelHandler has been initialized.')

    # ── private helpers ─────────────────────────────────────

    @property
    def _input_shape(self) -> Tuple[int, int, int]:
        return self.dataset_handler.input_shape

    @property
    def _class_num(self) -> int:
        return self.dataset_handler.class_num

    def _reg(self):
        """Return L2 regularizer from CONFIG, or None if weight_decay is 0."""
        wd = self.config['weight_decay']
        return regularizers.l2(wd) if wd > 0 else None

    def _head(self, x, dense_units: int, dropout_dense: float):
        x = GlobalAveragePooling2D()(x)
        x = Dense(dense_units, activation='relu', kernel_regularizer=self._reg())(x)
        x = Dropout(dropout_dense)(x)
        return Dense(self._class_num, activation='softmax')(x)

    def _freeze_bn(self, base_model) -> None:
        """
        Set all BatchNormalization layers to non-trainable.
        Prevents BN statistics from drifting when the backbone is partially frozen.
        """
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    def _apply_strategy(self, base_model, strategy: str, model_name: str) -> None:
        """
        Freeze or unfreeze backbone layers according to the transfer learning strategy.

        tl  - freeze entire backbone
        pft - freeze early layers, unfreeze later layers (architecture-specific cut-point)
        fft - unfreeze entire backbone
        """
        if strategy not in self.TRANSFER_STRATEGIES:
            raise ValueError(
                f"Strategy must be one of {self.TRANSFER_STRATEGIES}, got: '{strategy}'"
            )

        if strategy == 'tl':
            base_model.trainable = False

        elif strategy == 'pft':
            if model_name == 'VGG16':
                # Unfreeze block4 and block5 only
                for layer in base_model.layers:
                    layer.trainable = any(block in layer.name for block in self.VGG16_PFT_BLOCKS)
            else:
                name_key = model_name.lower()
                cut = self.PFT_LAYER_INDEX.get(name_key)

                if cut is None:
                    # Fallback: unfreeze last 30% of layers
                    cut = int(len(base_model.layers) * 0.7)
                    print(f"  No PFT cut-point for '{model_name}', "
                          f"using fallback: freeze first {cut}/{len(base_model.layers)} layers")

                for layer in base_model.layers[:cut]:
                    layer.trainable = False
                for layer in base_model.layers[cut:]:
                    layer.trainable = True

            self._freeze_bn(base_model)

        elif strategy == 'fft':
            base_model.trainable = True

    def _loss(self):
        """Return loss function with optional label smoothing from CONFIG."""
        ls = self.config['label_smoothing']
        smoothing = ls.get('value', 0.0) if ls.get('enabled', False) else 0.0
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

    # ── builders ────────────────────────────────────────────

    def _build_simple_cnn(self) -> tf.keras.Model:
        """Build the custom SimpleCNN baseline architecture."""
        dropout_conv  = self.config['dropout_conv']
        dropout_dense = self.config['dropout_dense']
        dense_units   = self.config['dense_units']
        lr            = self.config['learning_rate']
        reg           = self._reg()

        model = tf.keras.Sequential([
            tf.keras.Input(shape=self._input_shape),

            # Block 1
            Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(dropout_conv),

            # Block 2
            Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (3, 3), padding='same', kernel_regularizer=reg),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((2, 2)),
            Dropout(dropout_conv),

            # Head
            GlobalAveragePooling2D(),
            Dense(dense_units, activation='relu', kernel_regularizer=reg),
            Dropout(dropout_dense),
            Dense(self._class_num, activation='softmax'),
        ], name='SimpleCNN')

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=self._loss(),
            metrics=['accuracy']
        )
        return model

    def _build_transfer_model(self, base_model_cls, model_name: str) -> tf.keras.Model:
        """Unified builder for all transfer learning architectures."""
        strategy      = self.config['strategy']
        dropout_dense = self.config['dropout_dense']
        dense_units   = self.config['dense_units']
        lr            = self.config['learning_rate']

        base_model = base_model_cls(
            weights='imagenet',
            include_top=False,
            input_shape=self._input_shape,
        )

        self._apply_strategy(base_model, strategy, model_name)

        outputs = self._head(base_model.output, dense_units, dropout_dense)
        model   = Model(
            inputs=base_model.input,
            outputs=outputs,
            name=f'{model_name}_{strategy}',
        )

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=self._loss(),
            metrics=['accuracy'],
        )
        return model

    def _build_vgg16(self)          -> tf.keras.Model: return self._build_transfer_model(VGG16,          'VGG16')
    def _build_resnet50(self)       -> tf.keras.Model: return self._build_transfer_model(ResNet50,       'ResNet50')
    def _build_mobilenetv2(self)    -> tf.keras.Model: return self._build_transfer_model(MobileNetV2,    'MobileNetV2')
    def _build_efficientnetb0(self) -> tf.keras.Model: return self._build_transfer_model(EfficientNetB0, 'EfficientNetB0')

    # ── public ──────────────────────────────────────────────

    BUILDERS = {
        'SimpleCNN':      '_build_simple_cnn',
        'VGG16':          '_build_vgg16',
        'ResNet50':       '_build_resnet50',
        'MobileNetV2':    '_build_mobilenetv2',
        'EfficientNetB0': '_build_efficientnetb0',
    }

    def build(self) -> tf.keras.Model:
        """Build and return a compiled model based on current global CONFIG."""
        model_name = self.config['model']
        strategy   = self.config['strategy']

        if model_name not in self.BUILDERS:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available: {list(self.BUILDERS.keys())}"
            )

        if model_name == 'SimpleCNN' and strategy != 'baseline':
            raise ValueError(
                f"SimpleCNN only supports strategy='baseline', got: '{strategy}'"
            )

        if model_name != 'SimpleCNN' and strategy == 'baseline':
            raise ValueError(
                f"{model_name} requires a transfer strategy "
                f"{self.TRANSFER_STRATEGIES}, got: 'baseline'"
            )

        builder    = getattr(self, self.BUILDERS[model_name])
        self.model = builder()
        print(f'Model {model_name} ({strategy}) has been built.')
        return self.model

    def count_layers(self) -> dict:
        """Return trainable, frozen and total layer counts."""
        if self.model is None:
            print("No model built yet. Call build() first.")
            return {}

        trainable     = sum(1 for layer in self.model.layers if layer.trainable)
        non_trainable = sum(1 for layer in self.model.layers if not layer.trainable)

        return {
            'trainable':     trainable,
            'non_trainable': non_trainable,
            'total':         trainable + non_trainable,
        }

    def keras_summary(self) -> None:
        """Print Keras model summary."""
        if self.model is None:
            print("     No model built yet. Call build() first.")
            return
        self.model.summary()

    def count_params(self) -> dict:
        """Return trainable, non-trainable, total parameter counts and sizes in MB."""
        if self.model is None:
            print("No model built yet. Call build() first.")
            return {}

        trainable     = int(np.sum([np.prod(v.shape) for v in self.model.trainable_weights]))
        non_trainable = int(np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights]))
        total         = trainable + non_trainable

        def _to_mb(n): return round((n * 4) / (1024 * 1024), 2)

        return {
            'trainable':         trainable,
            'non_trainable':     non_trainable,
            'total':             total,
            'trainable_mb':      _to_mb(trainable),
            'non_trainable_mb':  _to_mb(non_trainable),
            'total_mb':          _to_mb(total),
        }

    def print_param_summary(self) -> None:
        """Print formatted parameter count summary."""
        if self.model is None:
            print("     No model built yet. Call build() first.")
            return

        params     = self.count_params()
        model_name = self.config['model']
        strategy   = self.config['strategy']

        print("=" * 45)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 45)
        print(f"Model:           {model_name}")
        print(f"Strategy:        {strategy}")
        print(f"Input shape:     {self._input_shape}")
        print(f"Output classes:  {self._class_num}")
        print(f"Learning rate:   {self.config['learning_rate']:.2e}")
        print("-" * 45)
        print(f"Trainable:       {params['trainable']:>12,}  ({params['trainable_mb']} MB)")
        print(f"Non-trainable:   {params['non_trainable']:>12,}  ({params['non_trainable_mb']} MB)")
        print(f"Total:           {params['total']:>12,}  ({params['total_mb']} MB)")
        print("=" * 45)

    def print_layer_trainability(self) -> None:
        """Print trainability status and parameter count for each layer."""
        if self.model is None:
            print("    No model built yet. Call build() first.")
            return

        trainable_count = 0
        frozen_count    = 0

        print(f"\n{'Layer':<50} {'Trainable':<10} {'Params':>10}")
        print("-" * 72)
        for layer in self.model.layers:
            params = layer.count_params()
            status = "yes" if layer.trainable else "no"
            print(f"{layer.name:<50} {status:<10} {params:>10,}")

            if layer.trainable:
                trainable_count += 1
            else:
                frozen_count += 1

        total = trainable_count + frozen_count
        print("-" * 72)
        print(f"Trainable: {trainable_count}/{total} | Frozen: {frozen_count}/{total}")

    def generate_summary(self, mode: str) -> None:
        params = self.count_params()
        layers = self.count_layers()

        summary_data = [
            ('Model',                self.config['model']),
            ('Strategy',             self.config['strategy']),
            None,
            ('Input shape',          str(self._input_shape)),
            ('Output classes',       self._class_num),
            None,
            ('Learning rate',        f"{self.config['learning_rate']:.2e}"),
            ('Dropout conv',         self.config['dropout_conv'] if self.config['model'] == 'SimpleCNN' else 'n/a'),
            ('Dropout dense',        self.config['dropout_dense']),
            ('Dense units',          self.config['dense_units']),
            ('Weight decay',         self.config['weight_decay']),
            ('Label smoothing',      f"{self.config['label_smoothing']['value']:.2f}" if self.config['label_smoothing']['enabled'] else 'off'),
            None,
            ('Trainable layers',     f"{layers['trainable']}/{layers['total']}"     if layers else 'n/a'),
            ('Frozen layers',        f"{layers['non_trainable']}/{layers['total']}" if layers else 'n/a'),
            None,
            ('Trainable params',     f"{params['trainable']:,} ({params['trainable_mb']} MB)"         if params else 'n/a'),
            ('Non-trainable params', f"{params['non_trainable']:,} ({params['non_trainable_mb']} MB)" if params else 'n/a'),
            ('Total params',         f"{params['total']:,} ({params['total_mb']} MB)"                 if params else 'n/a'),
        ]

        if mode == 'latex':
            self._generate_latex_summary('ModelHandler', summary_data, 'model_summary.tex')
        else:
            self._generate_ascii_summary('ModelHandler', summary_data)