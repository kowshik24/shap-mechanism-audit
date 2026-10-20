from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'dataset' / 'ai4i2020.csv'
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'experiments' / 'results'
FIGURES_DIR = ROOT / 'paper' / 'figures'
TABLES_DIR = ROOT / 'paper' / 'ieee_bigdata' / 'tables'

SEED = 42

MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
PHYSICAL = ['TWF', 'HDF', 'PWF', 'OSF']

RAW_NUMERIC = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
               'Torque [Nm]', 'Tool wear [min]']
DERIVED = ['Power [W]', 'Temp_diff [K]', 'Overstrain [minNm]', 'Overstrain_ratio']
ENGINEERED = RAW_NUMERIC + DERIVED

STRAIN_LIMIT = {'L': 11000, 'M': 12000, 'H': 13000}

# Features each mechanism's published trigger rule depends on (raw inputs and derived versions).
TRIGGER_FEATURES = {
    'TWF': ['Tool wear [min]'],
    'HDF': ['Air temperature [K]', 'Process temperature [K]', 'Temp_diff [K]', 'Rotational speed [rpm]'],
    'PWF': ['Torque [Nm]', 'Rotational speed [rpm]', 'Power [W]'],
    'OSF': ['Tool wear [min]', 'Torque [Nm]', 'Overstrain [minNm]', 'Overstrain_ratio'],
}

XGB_SEARCH_SPACE = {
    'n_estimators': [200, 300, 400, 600],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
}
XGB_SEARCH_ITER = 40
