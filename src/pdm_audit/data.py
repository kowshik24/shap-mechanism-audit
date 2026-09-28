import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, MODES, SEED, STRAIN_LIMIT


def load() -> pd.DataFrame:
    """Raw AI4I 2020 data with the four trigger-derived features added."""
    return add_features(pd.read_csv(DATA_PATH))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Overstrain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
    df['Overstrain_ratio'] = df['Overstrain [minNm]'] / df['Type'].map(STRAIN_LIMIT)
    return df


def split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 70/20/10 train/validation/test split; returns index arrays."""
    y = df['Machine failure']
    idx_train, idx_temp = train_test_split(df.index, test_size=0.30, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(idx_temp, test_size=1 / 3, stratify=y[idx_temp], random_state=seed)
    return idx_train, idx_val, idx_test


def label_facts(df: pd.DataFrame) -> dict:
    y, n_modes = df['Machine failure'], df[MODES].sum(axis=1)
    return {
        'rows': int(len(df)),
        'machine_failure_total': int(y.sum()),
        'failures_with_no_mode': int(((y == 1) & (n_modes == 0)).sum()),
        'rows_with_multiple_modes': int((n_modes > 1).sum()),
        'mode_counts': {m: int(df[m].sum()) for m in MODES},
        'RNF_with_machine_failure_0': int(((df['RNF'] == 1) & (y == 0)).sum()),
    }


def mechanism_groups(df: pd.DataFrame, exclusive: bool = False) -> dict:
    """Row indices where each mechanism fired (optionally only rows with that mechanism alone)."""
    n_modes = df[MODES].sum(axis=1)
    groups = {}
    for m in MODES:
        mask = df[m] == 1
        if exclusive:
            mask &= n_modes == 1
        groups[m] = df.index[mask].to_numpy()
    return groups
