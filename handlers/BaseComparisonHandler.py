import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .BaseHandler import BaseHandler


_CLASS_ALIASES = {
    'happy':     'Happiness',
    'happiness': 'Happiness',
    'sad':       'Sadness',
    'sadness':   'Sadness',
    'angry':     'Anger',
    'anger':     'Anger',
    'disgust':   'Disgust',
    'fear':      'Fear',
    'neutral':   'Neutral',
    'surprise':  'Surprise',
}


@dataclass
class ExperimentRecord:
    experiment_id: str
    model:    str
    dataset:  str
    strategy: str
    config:   dict = field(default_factory=dict)
    metrics:  dict = field(default_factory=dict)
    history:  dict = field(default_factory=dict)

    @property
    def test_accuracy(self) -> Optional[float]:
        return self.metrics.get('test_accuracy') or self.metrics.get('accuracy')

    @property
    def test_f1_macro(self) -> Optional[float]:
        return self.metrics.get('test_f1_macro') or self.metrics.get('f1_macro')

    @property
    def per_class_f1(self) -> dict:
        return self.metrics.get('per_class_f1', {})

    @property
    def tflite(self) -> dict:
        return self.metrics.get('tflite', {})


class BaseComparisonHandler(BaseHandler):
    """Shared base for Keras and TFLite comparison handlers — loading, DataFrame building, shared constants."""

    MODEL_ORDER = [
        'SimpleCNN', 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0',
    ]
    DATASET_ORDER = [
        'FER2013', 'CK+', 'RAF-DB', 'AffectNet',
    ]
    STRATEGY_ORDER = [
        'Baseline', 'TL', 'PFT', 'FFT',
    ]
    EMOTION_CLASSES = [
        'Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise',
    ]
    TFLITE_VARIANTS = [
        'float32', 'dynamic_quant', 'int8_quant',
    ]
    MODEL_COLORS = {
        'SimpleCNN':      '#4C72B0',
        'VGG16':          '#DD8452',
        'ResNet50':       '#55A868',
        'MobileNetV2':    '#C44E52',
        'EfficientNetB0': '#8172B3',
    }
    STRATEGY_LS = {
        'Baseline': '-',
        'TL':       '--',
        'PFT':      '-.',
        'FFT':      ':',
    }
    STRATEGY_COLORS = {
        'Baseline': '#4878CF',
        'TL':       '#6ACC65',
        'PFT':      '#D65F5F',
        'FFT':      '#B47CC7',
    }

    _FOLDER_RE = re.compile(
        r'^(?P<id>\d+)_(?P<dataset>[^_]+(?:[-_]DB)?)_(?P<model>[^_]+)_'
        r'(?P<strategy>[^_]+)_',
        re.IGNORECASE,
    )
    _STRATEGY_MAP = {
        'baseline': 'Baseline', 'tl': 'TL', 'pft': 'PFT', 'fft': 'FFT',
    }

    def __init__(
        self,
        train_experiments_dir: str,
        visualizations_directory: str,
    ):
        super().__init__(visualizations_directory)
        self.train_experiments_dir = Path(train_experiments_dir)
        self.records: list[ExperimentRecord] = []
        self.df: Optional[pd.DataFrame]      = None

        self._load()

    @staticmethod
    def _find_metrics_file(source: Path) -> Path:
        candidates = sorted(source.glob('metrics_*.json'))
        if candidates:
            return candidates[0]
        return source / 'metrics.json'

    def _load(self) -> None:
        folders = sorted([
            p for p in self.train_experiments_dir.iterdir()
            if p.is_dir() and self._FOLDER_RE.match(p.name)
        ])
        if not folders:
            raise FileNotFoundError(
                f'No experiment folders found in {self.train_experiments_dir}. '
                'Call ComparisonOrchestrator.unzip_experiments() first.'
            )
        print(f'Found {len(folders)} experiment folders — loading ...')
        failed = []
        for folder in folders:
            try:
                rec = self._load_folder(folder)
                if rec is not None:
                    self.records.append(rec)
            except Exception as exc:
                failed.append((folder.name, str(exc)))
        print(f'  loaded  {len(self.records)} experiments')
        if failed:
            print(f'  failed  {len(failed)}:')
            for name, err in failed:
                print(f'      {name}: {err}')
        self.df = self._build_dataframe()

    def summary(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError('No data loaded.')
        return self.df.copy()

    def _load_folder(self, folder: Path) -> Optional[ExperimentRecord]:
        """Load experiment data from a folder, handling possible nesting and missing files."""
        nested = folder / folder.name
        if nested.is_dir():
            folder = nested

        archive = folder / 'archive'
        source  = archive if archive.is_dir() else folder

        stem      = folder.name
        m         = self._FOLDER_RE.match(stem)
        from_name = {}
        if m:
            from_name = {
                'id':       m.group('id').lstrip('0') or '0',
                'dataset':  m.group('dataset'),
                'model':    m.group('model'),
                'strategy': self._STRATEGY_MAP.get(
                                m.group('strategy').lower(), m.group('strategy')),
            }

        config  = self._read_json(source / 'config.json')
        metrics = self._read_json(self._find_metrics_file(source))
        history = self._read_pickle(source / 'history.pkl')

        experiment_id = (metrics.get('experiment_id')
                         or config.get('experiment_id')
                         or from_name.get('id', stem))
        model    = (metrics.get('model')    or config.get('model_name')    or from_name.get('model',    'Unknown'))
        dataset  = (metrics.get('dataset')  or config.get('dataset_name')  or from_name.get('dataset',  'Unknown'))
        strategy = (metrics.get('strategy') or config.get('strategy')      or from_name.get('strategy', 'Unknown'))

        strategy = self._STRATEGY_MAP.get(strategy.lower(), strategy)

        # normalize dataset names
        if 'RAF' in dataset.upper():
            dataset = 'RAF-DB'
        elif 'AFFECT' in dataset.upper():
            dataset = 'AffectNet'

        raw_f1     = metrics.get('per_class_f1', {})
        normalized = {
            _CLASS_ALIASES.get(k.lower(), k): v
            for k, v in raw_f1.items()
        }
        metrics['per_class_f1'] = normalized

        return ExperimentRecord(
            experiment_id=str(experiment_id),
            model=model, dataset=dataset, strategy=strategy,
            config=config, metrics=metrics,
            history=history if isinstance(history, dict) else {},
        )

    def _build_dataframe(self) -> pd.DataFrame:
        """Build dataframe for each trained experiment."""
        rows = []
        for r in self.records:
            val_acc   = r.history.get('val_accuracy', [])
            train_acc = r.history.get('accuracy', [])

            best_val_acc_from_history    = float(max(val_acc))   if val_acc   else None
            actual_epochs_from_history   = len(val_acc)          if val_acc   else None
            final_train_acc_from_history = float(train_acc[-1])  if train_acc else None
            train_val_gap_from_history   = (
                round(float(train_acc[-1]) - float(val_acc[-1]), 6)
                if train_acc and val_acc else None
            )

            row = {
                'experiment_id':     r.experiment_id,
                'model':             r.model,
                'dataset':           r.dataset,
                'strategy':          r.strategy,
                'test_accuracy':     r.test_accuracy,
                'test_f1_macro':     r.test_f1_macro,
                'precision_macro':   r.metrics.get('precision_macro'),
                'recall_macro':      r.metrics.get('recall_macro'),
                'macro_auc':         r.metrics.get('macro_auc'),
                'actual_epochs':     r.metrics.get('actual_epochs')        or actual_epochs_from_history,
                'best_epoch':        r.metrics.get('best_epoch'),
                'best_val_accuracy': r.metrics.get('best_val_accuracy')    or best_val_acc_from_history,
                'best_val_loss':     r.metrics.get('best_val_loss'),
                'train_val_gap':     r.metrics.get('train_val_gap')        or train_val_gap_from_history,
                'final_train_acc':   r.metrics.get('final_train_accuracy') or final_train_acc_from_history,
                'early_stopping':    r.metrics.get('early_stopping_triggered'),
                'ece':               r.metrics.get('ece'),
                'brier_score':       r.metrics.get('brier_score'),
                'conf_correct':      r.metrics.get('confidence_mean_correct'),
                'conf_incorrect':    r.metrics.get('confidence_mean_incorrect'),
            }

            for emo in self.EMOTION_CLASSES:
                row[f'f1_{emo.lower()}'] = r.per_class_f1.get(emo)

            for vkey in self.TFLITE_VARIANTS:
                vdata = r.tflite.get(vkey, {})
                row[f'tflite_{vkey}_accuracy']          = vdata.get('accuracy')
                row[f'tflite_{vkey}_size_kb']           = vdata.get('model_size_kb')
                row[f'tflite_{vkey}_latency_ms']        = vdata.get('mean_inference_time_ms')
                row[f'tflite_{vkey}_p95_ms']            = vdata.get('p95_inference_time_ms')
                row[f'tflite_{vkey}_compression_ratio'] = vdata.get('compression_ratio')
                row[f'tflite_{vkey}_accuracy_delta']    = vdata.get('accuracy_delta_vs_keras')
                row[f'tflite_{vkey}_f1_macro']          = vdata.get('f1_macro')

            rows.append(row)

        df = pd.DataFrame(rows)
        for col, order in [
            ('model',    self.MODEL_ORDER),
            ('dataset',  self.DATASET_ORDER),
            ('strategy', self.STRATEGY_ORDER),
        ]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=order, ordered=True)
        return df.sort_values(['model', 'dataset', 'strategy']).reset_index(drop=True)

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _read_pickle(path: Path):
        if not path.exists():
            return {}
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _check_loaded(self) -> None:
        if self.df is None:
            raise RuntimeError('No data loaded.')

    def print_summary(self, mode: str = 'ascii') -> None:
        if self.df is None:
            print('No data loaded.')
            return
        sections = [
            ('Experiments loaded', len(self.records)),
            ('Models',             ', '.join(str(m) for m in self.df['model'].unique())),
            ('Datasets',           ', '.join(str(d) for d in self.df['dataset'].unique())),
            ('Strategies',         ', '.join(str(s) for s in self.df['strategy'].unique())),
        ]
        if mode == 'latex':
            self._generate_latex_summary(self.__class__.__name__, sections,
                                         'comparison_summary.tex')
        else:
            self._generate_ascii_summary(self.__class__.__name__, sections)
