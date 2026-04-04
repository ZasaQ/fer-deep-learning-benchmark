import json
import ipywidgets as w
from IPython.display import display, clear_output, HTML

# ──────────────────────────────────────────────
# PRESETS
# ──────────────────────────────────────────────
AUGMENTATION_PRESETS = {
    'Conservative':    {'rotation_range': 10,  'width_shift_range': 0.05, 'height_shift_range': 0.05,
                        'zoom_range': 0.1,  'brightness_range': [0.9, 1.1],  'shear_range': 0.0,
                        'horizontal_flip': True, 'fill_mode': 'constant'},
    'Medium':          {'rotation_range': 20,  'width_shift_range': 0.08, 'height_shift_range': 0.08,
                        'zoom_range': 0.1,  'brightness_range': [0.85, 1.15], 'shear_range': 0.0,
                        'horizontal_flip': True, 'fill_mode': 'constant'},
    'Aggressive':      {'rotation_range': 30,  'width_shift_range': 0.1,  'height_shift_range': 0.1,
                        'zoom_range': 0.15, 'brightness_range': [0.8, 1.2],   'shear_range': 0.0,
                        'horizontal_flip': True, 'fill_mode': 'constant'},
    'Very Aggressive': {'rotation_range': 45,  'width_shift_range': 0.15, 'height_shift_range': 0.15,
                        'zoom_range': 0.2,  'brightness_range': [0.7, 1.3],   'shear_range': 0.15,
                        'horizontal_flip': True, 'fill_mode': 'constant'},
}

DATASET_DEFAULT_PRESETS = {
    'FER2013':   'Aggressive',
    'CK+':       'Conservative',
    'RAF-DB':    'Medium',
    'AffectNet': 'Medium',
}

CONFIG = {}

# ──────────────────────────────────────────────
# BUILD CONFIG DICT
# ──────────────────────────────────────────────
def build_config(widgets):
    global CONFIG
    W = widgets
    CONFIG = {
        'dataset':     W['dataset'].value,
        'batch_size':  W['batch_size'].value,
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
    }
    return CONFIG

# ──────────────────────────────────────────────
# STYLE HELPERS
# ──────────────────────────────────────────────
_LABEL_STYLE = {'description_width': '150px'}
_LAYOUT      = w.Layout(width='320px')

def _slider(label, mn, mx, val, step, fmt=None):
    kw = dict(min=mn, max=mx, value=val, step=step,
              description=label, style=_LABEL_STYLE, layout=_LAYOUT)
    if fmt:
        kw['readout_format'] = fmt
    return w.FloatSlider(**kw) if isinstance(step, float) else w.IntSlider(**kw)

def _dropdown(label, opts, val):
    return w.Dropdown(options=opts, value=val, description=label,
                      style=_LABEL_STYLE, layout=_LAYOUT)

def _checkbox(label, val):
    return w.Checkbox(value=val, description=label,
                      style={'description_width': 'initial'}, indent=False, layout=_LAYOUT)

# ──────────────────────────────────────────────
# BUILD ALL WIDGETS
# ──────────────────────────────────────────────
def _make_widgets():
    W = {}
    _p = AUGMENTATION_PRESETS[DATASET_DEFAULT_PRESETS['FER2013']]

    W['dataset']         = _dropdown('Dataset:', ['FER2013', 'CK+', 'RAF-DB', 'AffectNet'], 'FER2013')
    W['batch_size']      = _slider('Batch size:', 8, 128, 32, 8)
    W['use_augmentation']= _checkbox('Enable Augmentation', True)
    W['aug_preset']      = _dropdown('Preset:', list(AUGMENTATION_PRESETS.keys()) + ['Custom'],
                                     DATASET_DEFAULT_PRESETS['FER2013'])
    W['rotation']        = _slider('Rotation:',      0,   50,  _p['rotation_range'],        5)
    W['width_shift']     = _slider('Width shift:',   0.0, 0.3, _p['width_shift_range'],     0.05, '.2f')
    W['height_shift']    = _slider('Height shift:',  0.0, 0.3, _p['height_shift_range'],    0.05, '.2f')
    W['zoom']            = _slider('Zoom:',           0.0, 0.3, _p['zoom_range'],            0.05, '.2f')
    W['brightness_min']  = _slider('Brightness min:',0.5, 1.0, _p['brightness_range'][0],   0.05, '.2f')
    W['brightness_max']  = _slider('Brightness max:',1.0, 1.5, _p['brightness_range'][1],   0.05, '.2f')
    W['shear']           = _slider('Shear range:',   0.0, 0.3, _p['shear_range'],           0.05, '.2f')
    W['horizontal_flip'] = _checkbox('Horizontal flip', _p['horizontal_flip'])
    W['fill_mode']       = _dropdown('Fill mode:', ['constant', 'nearest', 'reflect', 'wrap'], _p['fill_mode'])

    return W

# ──────────────────────────────────────────────
# MAIN UI FUNCTION
# ──────────────────────────────────────────────
def display_config():
    W = _make_widgets()

    out = w.Output()

    def refresh(_=None):
        cfg = build_config(W)
        with out:
            clear_output(wait=True)
            print(json.dumps(cfg, indent=2))

    # ── Augmentation visibility ──
    _aug_param_keys = ['aug_preset', 'rotation', 'width_shift', 'height_shift', 'zoom',
                       'brightness_min', 'brightness_max', 'shear', 'horizontal_flip', 'fill_mode']

    def _on_aug_toggle(change):
        display_val = '' if change['new'] else 'none'
        for k in _aug_param_keys:
            W[k].layout.display = display_val
        refresh()

    # ── Preset logic ──
    _preset_applying = [False]

    def _on_preset_change(change):
        name = change['new']
        if name == 'Custom' or name not in AUGMENTATION_PRESETS:
            return
        _preset_applying[0] = True
        p = AUGMENTATION_PRESETS[name]
        W['rotation'].value        = p['rotation_range']
        W['width_shift'].value     = p['width_shift_range']
        W['height_shift'].value    = p['height_shift_range']
        W['zoom'].value            = p['zoom_range']
        W['brightness_min'].value  = p['brightness_range'][0]
        W['brightness_max'].value  = p['brightness_range'][1]
        W['shear'].value           = p['shear_range']
        W['horizontal_flip'].value = p['horizontal_flip']
        W['fill_mode'].value       = p['fill_mode']
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

    # ── Wire observers ──
    W['use_augmentation'].observe(_on_aug_toggle, names='value')
    W['aug_preset'].observe(_on_preset_change, names='value')
    for k in ['rotation', 'width_shift', 'height_shift', 'zoom',
              'brightness_min', 'brightness_max', 'shear', 'horizontal_flip', 'fill_mode']:
        W[k].observe(_on_aug_manual_change, names='value')

    for key in ['dataset', 'batch_size', 'use_augmentation', 'aug_preset', 'rotation', 'width_shift',
                'height_shift', 'zoom', 'brightness_min', 'brightness_max',
                'shear', 'horizontal_flip', 'fill_mode']:
        W[key].observe(refresh, names='value')

    # ── Layout ──
    display(HTML("""
    <style>
        .widget-label { text-align: left !important; min-width: 150px !important; }
        .widget-hbox  { align-items: flex-start !important; }
        .widget-vbox  { align-items: flex-start !important; }
        .widget-checkbox { width: auto !important; justify-content: flex-start !important; }
        .widget-checkbox input[type="checkbox"] { margin-left: 0 !important; margin-right: 8px !important; }
    </style>
    """))

    aug_params = w.VBox(
        [W['aug_preset'], W['rotation'], W['width_shift'], W['height_shift'], W['zoom'],
         W['brightness_min'], W['brightness_max'], W['shear'], W['horizontal_flip'], W['fill_mode']],
        layout=w.Layout(align_items='flex-start')
    )

    ui = w.VBox([
        w.HTML('<h2>Dataset & Augmentation Config</h2>'),
        W['dataset'],
        W['batch_size'],
        w.HTML("<hr style='width:320px; margin: 10px 0'>"),
        w.HTML('<h3>Data Augmentation</h3>'),
        W['use_augmentation'],
        aug_params,
        w.HTML("<hr style='width:320px; margin: 10px 0'>"),
        w.HTML('<b style="color:#aaa;font-size:12px">Current Config (live)</b>'),
        out,
    ], layout=w.Layout(align_items='flex-start', width='400px'))

    display(ui)
    refresh()