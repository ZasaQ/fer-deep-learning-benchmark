import re
import json
import ipywidgets as w
from IPython.display import display, clear_output, Javascript

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def format_config_id(value):
    return f'{int(value):02d}'

def parse_float(val, default):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def format_float(val):
    """Always return scientific notation."""
    try:
        f = float(val)
        s = f'{f:.0e}'
        s = re.sub(r'e([+-])0+(\d)', r'e\1\2', s)
        return s
    except (ValueError, TypeError):
        return str(val)

AUGMENTATION_PRESETS = {
    'Conservative':   {'rotation_range': 10,  'width_shift_range': 0.05, 'height_shift_range': 0.05,
                       'zoom_range': 0.1,  'brightness_range': [0.9, 1.1],  'shear_range': 0.0,
                       'horizontal_flip': True, 'fill_mode': 'constant'},
    'Medium':         {'rotation_range': 20,  'width_shift_range': 0.08, 'height_shift_range': 0.08,
                       'zoom_range': 0.1,  'brightness_range': [0.85, 1.15], 'shear_range': 0.0,
                       'horizontal_flip': True, 'fill_mode': 'constant'},
    'Aggressive':     {'rotation_range': 30,  'width_shift_range': 0.1,  'height_shift_range': 0.1,
                       'zoom_range': 0.15, 'brightness_range': [0.8, 1.2],   'shear_range': 0.0,
                       'horizontal_flip': True, 'fill_mode': 'constant'},
    'Very Aggressive':{'rotation_range': 45,  'width_shift_range': 0.15, 'height_shift_range': 0.15,
                       'zoom_range': 0.2,  'brightness_range': [0.7, 1.3],   'shear_range': 0.15,
                       'horizontal_flip': True, 'fill_mode': 'constant'},
}

CONFIG = {}

# ──────────────────────────────────────────────
# BUILD CONFIG DICT
# ──────────────────────────────────────────────
def build_config(widgets):
    global CONFIG
    W = widgets
    CONFIG = {
        'id':            format_config_id(W['config_id'].value),
        'dataset':       W['dataset'].value,
        'model':         W['model'].value,
        'strategy':      W['strategy'].value,
        'learning_rate': parse_float(W['learning_rate'].value, 1e-3),
        'batch_size':    W['batch_size'].value,
        'epochs':        W['epochs'].value,
        'dropout_conv':  W['dropout_conv'].value if W['model'].value == 'SimpleCNN' else None,
        'dropout_dense': W['dropout_dense'].value,
        'weight_decay':  parse_float(W['weight_decay'].value, 1e-4),
        'dense_units':   W['dense_units'].value,
        'class_weights': {
            'enabled': W['use_class_weights'].value,
            'mode':    W['class_weights_mode'].value if W['use_class_weights'].value else None,
        },
        'label_smoothing': {
            'enabled': W['use_label_smoothing'].value,
            'value':   W['label_smoothing_value'].value if W['use_label_smoothing'].value else 0.0,
        },
        'augmentation': {
            'enabled':            W['use_augmentation'].value,
            'preset':             W['aug_preset'].value,
            'rotation_range':     W['rotation'].value,
            'width_shift_range':  W['width_shift'].value,
            'height_shift_range': W['height_shift'].value,
            'zoom_range':         W['zoom'].value,
            'brightness_range':   [W['brightness_min'].value, W['brightness_max'].value],
            'shear_range':        W['shear'].value,
            'horizontal_flip':    W['horizontal_flip'].value,
            'fill_mode':          W['fill_mode'].value,
        },
        'callbacks': {
            'early_stopping': {
                'enabled':   W['use_early_stopping'].value,
                'patience':  W['es_patience'].value,
                'min_delta': parse_float(W['es_min_delta'].value, 1e-4),
            },
            'reduce_lr': {
                'enabled':  W['use_reduce_lr'].value,
                'patience': W['rlr_patience'].value,
                'factor':   W['rlr_factor'].value,
                'min_lr':   parse_float(W['rlr_min_lr'].value, 1e-7),
            },
        },
    }
    return CONFIG

# ──────────────────────────────────────────────
# STYLE HELPERS
# ──────────────────────────────────────────────
_LABEL_STYLE  = {'description_width': '160px'}
_SLIDER_LAYOUT = w.Layout(width='340px')
_TEXT_LAYOUT   = w.Layout(width='300px')
_DROP_LAYOUT   = w.Layout(width='300px')

def _slider(label, mn, mx, val, step, fmt=None):
    kw = dict(min=mn, max=mx, value=val, step=step,
              description=label, style=_LABEL_STYLE, layout=w.Layout(width='340px'))
    if fmt:
        kw['readout_format'] = fmt
    return w.FloatSlider(**kw) if isinstance(step, float) else w.IntSlider(**kw)

def _textbox(label, val):
    return w.Text(value=str(val), description=label,
                  style=_LABEL_STYLE, layout=w.Layout(width='300px'))

def _dropdown(label, opts, val):
    return w.Dropdown(options=opts, value=val, description=label,
                      style=_LABEL_STYLE, layout=w.Layout(width='300px'))

def _checkbox(label, val):
    return w.Checkbox(value=val, description=label,
                      style=_LABEL_STYLE, indent=False)

def _section(title):
    return w.HTML(f'<b style="font-size:13px;color:#aaa">{title}</b>')

def _hr():
    return w.HTML('<hr style="border-color:#444;margin:6px 0">')

# ──────────────────────────────────────────────
# BUILD ALL WIDGETS
# ──────────────────────────────────────────────
def _make_widgets():
    W = {}

    W['config_id']      = w.BoundedIntText(value=1, min=1, max=99, description='Experiment ID',
                                           style=_LABEL_STYLE, layout=_TEXT_LAYOUT)
    W['dataset']        = _dropdown('Dataset',  ['FER2013', 'CK+', 'RAF-DB', 'AffectNet'], 'FER2013')
    W['model']          = _dropdown('Model',    ['SimpleCNN', 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0'], 'SimpleCNN')

    ALL_STRATEGIES = [('Baseline', 'baseline')]
    TL_STRATEGIES  = [('Transfer learning', 'tl'), ('Partial fine-tuning', 'pft'), ('Full fine-tuning', 'fft')]
    W['_ALL_STRATEGIES'] = ALL_STRATEGIES
    W['_TL_STRATEGIES']  = TL_STRATEGIES
    W['strategy']       = _dropdown('Strategy', ALL_STRATEGIES, 'baseline')

    W['learning_rate']  = _textbox('Learning rate', '1e-3')
    W['batch_size']     = _slider('Batch size',    8,   128,  32,   8)
    W['epochs']         = _slider('Epochs',        10,  200,  50,   10)
    W['dropout_conv']   = _slider('Dropout (conv)',0.0, 0.7,  0.25, 0.05, '.2f')
    W['dropout_dense']  = _slider('Dropout (dense)',0.0,0.8,  0.5,  0.05, '.2f')
    W['dense_units']    = _slider('Dense units',   64,  512,  256,  64)
    W['weight_decay']   = _textbox('Weight decay', '1e-4')

    W['use_class_weights']    = _checkbox('Use Class Weights', True)
    W['class_weights_mode']   = _dropdown('Mode', [('Balanced (auto)', 'balanced'), ('Manual', 'manual')], 'balanced')
    W['use_label_smoothing']  = _checkbox('Label Smoothing', True)
    W['label_smoothing_value']= _slider('Smoothing', 0.0, 0.2, 0.1, 0.01, '.2f')

    # Augmentation
    W['use_augmentation'] = _checkbox('Enable Augmentation', True)
    W['aug_preset']       = _dropdown('Preset',
                                      ['Conservative', 'Medium', 'Aggressive', 'Very Aggressive', 'Custom'],
                                      'Aggressive')
    W['rotation']         = _slider('Rotation range',  0,   50,  30,   5)
    W['width_shift']      = _slider('Width shift',     0.0, 0.3, 0.1,  0.05, '.2f')
    W['height_shift']     = _slider('Height shift',    0.0, 0.3, 0.1,  0.05, '.2f')
    W['zoom']             = _slider('Zoom',            0.0, 0.3, 0.15, 0.05, '.2f')
    W['brightness_min']   = _slider('Brightness min',  0.5, 1.0, 0.8,  0.05, '.2f')
    W['brightness_max']   = _slider('Brightness max',  1.0, 1.5, 1.2,  0.05, '.2f')
    W['shear']            = _slider('Shear range',     0.0, 0.3, 0.0,  0.05, '.2f')
    W['horizontal_flip']  = _checkbox('Horizontal flip', True)
    W['fill_mode']        = _dropdown('Fill mode', ['constant', 'nearest', 'reflect', 'wrap'], 'constant')

    # Callbacks
    W['use_early_stopping'] = _checkbox('Early Stopping', True)
    W['es_patience']        = _slider('ES patience',  5,  30, 15, 5)
    W['es_min_delta']       = _textbox('ES min delta', '1e-4')
    W['use_reduce_lr']      = _checkbox('Reduce LR', True)
    W['rlr_patience']       = _slider('RLR patience', 2,  15,  5, 1)
    W['rlr_factor']         = _slider('RLR factor',   0.1,0.9, 0.5, 0.1, '.1f')
    W['rlr_min_lr']         = _textbox('RLR min LR',  '1e-7')

    return W

# ──────────────────────────────────────────────
# MAIN UI FUNCTION
# ──────────────────────────────────────────────
def display_config():
    W = _make_widgets()

    # ── Output area for live JSON preview ──
    out = w.Output()

    def refresh(_=None):
        cfg = build_config(W)
        with out:
            clear_output(wait=True)
            print(json.dumps(cfg, indent=2))

    # ── Visibility observers ──
    def _on_model_change(change):
        is_simple = change['new'] == 'SimpleCNN'
        W['dropout_conv'].layout.display = '' if is_simple else 'none'
        strats = W['_ALL_STRATEGIES'] if is_simple else W['_TL_STRATEGIES']
        W['strategy'].options = strats
        W['strategy'].value   = 'baseline' if is_simple else 'tl'
        refresh()

    def _on_aug_toggle(change):
        display_val = '' if change['new'] else 'none'
        for k in ['aug_preset','rotation','width_shift','height_shift','zoom',
                  'brightness_min','brightness_max','shear','horizontal_flip','fill_mode']:
            W[k].layout.display = display_val
        refresh()

    def _on_es_toggle(change):
        for k in ['es_patience', 'es_min_delta']:
            W[k].layout.display = '' if change['new'] else 'none'
        refresh()

    def _on_rlr_toggle(change):
        for k in ['rlr_patience', 'rlr_factor', 'rlr_min_lr']:
            W[k].layout.display = '' if change['new'] else 'none'
        refresh()

    def _on_cw_toggle(change):
        W['class_weights_mode'].layout.display = '' if change['new'] else 'none'
        refresh()

    def _on_ls_toggle(change):
        W['label_smoothing_value'].layout.display = '' if change['new'] else 'none'
        refresh()

    # ── Augmentation preset handler ──
    _aug_manual_keys = ['rotation','width_shift','height_shift','zoom',
                        'brightness_min','brightness_max','shear','horizontal_flip','fill_mode']
    _preset_applying = [False]

    def _on_preset_change(change):
        name = change['new']
        if name == 'Custom' or name not in AUGMENTATION_PRESETS:
            return
        _preset_applying[0] = True
        p = AUGMENTATION_PRESETS[name]
        W['rotation'].value       = p['rotation_range']
        W['width_shift'].value    = p['width_shift_range']
        W['height_shift'].value   = p['height_shift_range']
        W['zoom'].value           = p['zoom_range']
        W['brightness_min'].value = p['brightness_range'][0]
        W['brightness_max'].value = p['brightness_range'][1]
        W['shear'].value          = p['shear_range']
        W['horizontal_flip'].value= p['horizontal_flip']
        W['fill_mode'].value      = p['fill_mode']
        _preset_applying[0] = False
        refresh()

    def _on_aug_manual_change(_change):
        if _preset_applying[0]:
            return
        current = {
            'rotation_range':     W['rotation'].value,
            'width_shift_range':  W['width_shift'].value,
            'height_shift_range': W['height_shift'].value,
            'zoom_range':         W['zoom'].value,
            'brightness_range':   [W['brightness_min'].value, W['brightness_max'].value],
            'shear_range':        W['shear'].value,
            'horizontal_flip':    W['horizontal_flip'].value,
            'fill_mode':          W['fill_mode'].value,
        }
        for name, preset in AUGMENTATION_PRESETS.items():
            if all(current.get(k) == v for k, v in preset.items()):
                _preset_applying[0] = True
                W['aug_preset'].value = name
                _preset_applying[0] = False
                break
        else:
            _preset_applying[0] = True
            W['aug_preset'].value = 'Custom'
            _preset_applying[0] = False
        refresh()

    # Wire up observers
    W['model'].observe(_on_model_change, names='value')
    W['use_augmentation'].observe(_on_aug_toggle, names='value')
    W['use_early_stopping'].observe(_on_es_toggle, names='value')
    W['use_reduce_lr'].observe(_on_rlr_toggle, names='value')
    W['use_class_weights'].observe(_on_cw_toggle, names='value')
    W['use_label_smoothing'].observe(_on_ls_toggle, names='value')
    W['aug_preset'].observe(_on_preset_change, names='value')
    for k in _aug_manual_keys:
        W[k].observe(_on_aug_manual_change, names='value')

    for key in ['config_id','dataset','strategy','learning_rate','batch_size','epochs',
                'dropout_conv','dropout_dense','dense_units','weight_decay',
                'use_class_weights','class_weights_mode','use_label_smoothing',
                'label_smoothing_value','use_augmentation','aug_preset',
                'use_early_stopping','es_patience','es_min_delta',
                'use_reduce_lr','rlr_patience','rlr_factor','rlr_min_lr']:
        W[key].observe(refresh, names='value')

    # Preset JSON loader
    _btn_layout   = w.Layout(width='auto', height='28px')
    preset_upload = w.FileUpload(
        accept='.json', multiple=False,
        description='Load Config',
        layout=_btn_layout,
    )
    preset_status = w.Label(value='')

    def _on_preset_upload(change):
        if not change['new']:
            return
        try:
            content = list(change['new'].values())[0]['content']
            cfg = json.loads(content)

            def _g(key, default):
                v = cfg.get(key)
                return v if v is not None else default

            aug  = cfg.get('augmentation', {})
            cw   = cfg.get('class_weights', {})
            ls   = cfg.get('label_smoothing', {})
            es   = cfg.get('callbacks', {}).get('early_stopping', {})
            rlr  = cfg.get('callbacks', {}).get('reduce_lr', {})
            raw_id = cfg.get('id', 1)
            parsed_id = int(str(raw_id).lstrip('0') or '1')

            loaded_model    = _g('model', 'SimpleCNN')
            loaded_strategy = _g('strategy', 'baseline')
            is_simple = loaded_model == 'SimpleCNN'
            W['model'].value   = loaded_model
            strats = W['_ALL_STRATEGIES'] if is_simple else W['_TL_STRATEGIES']
            W['strategy'].options = strats
            if not is_simple and loaded_strategy == 'baseline':
                loaded_strategy = 'tl'
            W['strategy'].value = loaded_strategy

            W['config_id'].value      = parsed_id
            W['dataset'].value        = _g('dataset', 'FER2013')
            W['learning_rate'].value  = format_float(_g('learning_rate', 1e-3))
            W['batch_size'].value     = _g('batch_size', 32)
            W['epochs'].value         = _g('epochs', 50)
            W['dropout_conv'].value   = _g('dropout_conv', 0.25) or 0.25
            W['dropout_dense'].value  = _g('dropout_dense', 0.5)
            W['dense_units'].value    = _g('dense_units', 256)
            W['weight_decay'].value   = format_float(_g('weight_decay', 1e-4))
            W['use_class_weights'].value     = cw.get('enabled', True)
            W['class_weights_mode'].value    = cw.get('mode', 'balanced')
            W['use_label_smoothing'].value   = ls.get('enabled', True)
            W['label_smoothing_value'].value = ls.get('value', 0.1)

            W['use_augmentation'].value = aug.get('enabled', True)
            aug_preset_val = 'Custom'
            for name, vals in AUGMENTATION_PRESETS.items():
                if all(aug.get(k) == v for k, v in vals.items()):
                    aug_preset_val = name
                    break
            _preset_applying[0] = True
            W['aug_preset'].value      = aug_preset_val
            W['rotation'].value        = aug.get('rotation_range', 30)
            W['width_shift'].value     = aug.get('width_shift_range', 0.1)
            W['height_shift'].value    = aug.get('height_shift_range', 0.1)
            W['zoom'].value            = aug.get('zoom_range', 0.15)
            br = aug.get('brightness_range', [0.8, 1.2])
            W['brightness_min'].value  = br[0]
            W['brightness_max'].value  = br[1]
            W['shear'].value           = aug.get('shear_range', 0.0)
            W['horizontal_flip'].value = aug.get('horizontal_flip', True)
            W['fill_mode'].value       = aug.get('fill_mode', 'constant')
            _preset_applying[0] = False

            W['use_early_stopping'].value = es.get('enabled', True)
            W['es_patience'].value        = es.get('patience', 15)
            W['es_min_delta'].value       = format_float(es.get('min_delta', 1e-4))
            W['use_reduce_lr'].value      = rlr.get('enabled', True)
            W['rlr_patience'].value       = rlr.get('patience', 5)
            W['rlr_factor'].value         = rlr.get('factor', 0.5)
            W['rlr_min_lr'].value         = format_float(rlr.get('min_lr', 1e-7))

            preset_status.value = 'Preset loaded!'
            refresh()
        except Exception as e:
            preset_status.value = f'Error while loading preset: {e}'

    preset_upload.observe(_on_preset_upload, names='value')

    # Save config button
    save_btn = w.Button(
        description='Save Current Config',
        button_style='',
        layout=_btn_layout,
        tooltip='Download current config as JSON file',
    )
    save_status = w.Label(value='')

    def _on_save_click(_):
        cfg = build_config(W)
        filename = f'config_{cfg["id"]}_{cfg["dataset"]}_{cfg["model"]}_{cfg["strategy"]}.json'
        json_str = json.dumps(cfg, indent=2)

        js_code = f"""
        (function() {{
            var data = {json.dumps(json_str)};
            var blob = new Blob([data], {{type: 'application/json'}});
            var url  = URL.createObjectURL(blob);
            var a    = document.createElement('a');
            a.href     = url;
            a.download = {json.dumps(filename)};
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }})();
        """
        display(Javascript(js_code))
        save_status.value = f'Config saved: {filename}'

    save_btn.on_click(_on_save_click)

    # ──────────────────────────────────────────────
    # LAYOUT
    # ──────────────────────────────────────────────
    col_layout = w.Layout(
        width='370px', padding='8px 12px',
        border='1px solid #444', margin='4px'
    )

    col_left = w.VBox([
        _section('Dataset & Training'),
        W['config_id'], W['dataset'], W['model'], W['strategy'],
        W['learning_rate'], W['batch_size'], W['epochs'],
        W['dropout_conv'], W['dropout_dense'], W['dense_units'], W['weight_decay'],
        _hr(),
        W['use_class_weights'], W['class_weights_mode'],
        W['use_label_smoothing'], W['label_smoothing_value'],
    ], layout=col_layout)

    col_mid = w.VBox([
        _section('Data Augmentation'),
        W['use_augmentation'], W['aug_preset'],
        W['rotation'], W['width_shift'], W['height_shift'], W['zoom'],
        W['brightness_min'], W['brightness_max'], W['shear'],
        W['horizontal_flip'], W['fill_mode'],
    ], layout=col_layout)

    col_right = w.VBox([
        _section('Callbacks'),
        W['use_early_stopping'],
        W['es_patience'], W['es_min_delta'],
        _hr(),
        W['use_reduce_lr'],
        W['rlr_patience'], W['rlr_factor'], W['rlr_min_lr'],
    ], layout=col_layout)

    header = w.HBox([
        w.HTML('<h3 style="margin:0;color:#eee">Experiment Configurator</h3>'),
    ])

    loader_row = w.HBox(
        [preset_upload, preset_status, save_btn, save_status],
        layout=w.Layout(align_items='center', gap='12px', margin='6px 0')
    )

    columns    = w.HBox([col_left, col_mid, col_right],
                        layout=w.Layout(flex_flow='row wrap'))
    json_label = w.HTML('<b style="color:#aaa;font-size:12px">Current Config (live)</b>')

    ui = w.VBox([
        header,
        loader_row,
        columns,
        _hr(),
        json_label,
        out,
    ], layout=w.Layout(padding='12px', background='#1e1e1e'))

    display(ui)
    refresh()