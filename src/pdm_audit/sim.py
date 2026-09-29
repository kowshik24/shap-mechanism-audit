"""
AI4I 2020 re-simulated at arbitrary size.

Process variables are drawn jointly from the released rows (so their joint distribution and
correlations are kept) with small Gaussian jitter; failure flags are then recomputed from the
published trigger rules [Matzka 2020]. On the released data the HDF, PWF and OSF rules reproduce the
flags exactly. TWF is stochastic in the original simulator (failure at a random point of the
200-240 min tool-life window), so it is drawn with the probability that matches the released TWF
rate among worn tools. RNF fires with probability 0.001. Unlike the release, the simulated labels
are consistent: Machine failure is exactly the OR of the five flags.
"""
import numpy as np
import pandas as pd

from .config import STRAIN_LIMIT
from .data import add_features, load

JITTER = {'Air temperature [K]': 0.1, 'Process temperature [K]': 0.1,
          'Rotational speed [rpm]': 5.0, 'Torque [Nm]': 0.5}
TWF_MIN_WEAR = 198  # smallest tool wear at which the release records a TWF


def trigger_flags(df, rng):
    lim = df['Type'].map(STRAIN_LIMIT)
    power = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 2 * np.pi / 60
    out = pd.DataFrame(index=df.index)
    out['HDF'] = ((df['Process temperature [K]'] - df['Air temperature [K]'] < 8.6)
                  & (df['Rotational speed [rpm]'] < 1380)).astype(int)
    out['PWF'] = ((power < 3500) | (power > 9000)).astype(int)
    out['OSF'] = (df['Tool wear [min]'] * df['Torque [Nm]'] > lim).astype(int)
    return out


def simulate(n, seed=0):
    real = load()
    rng = np.random.default_rng(seed)
    q_twf = real['TWF'].sum() / (real['Tool wear [min]'] >= TWF_MIN_WEAR).sum()

    cols = ['Type', *JITTER, 'Tool wear [min]']
    df = real[cols].iloc[rng.integers(0, len(real), n)].reset_index(drop=True)
    for c, sd in JITTER.items():
        df[c] = df[c] + rng.normal(0, sd, n)
    df['Torque [Nm]'] = df['Torque [Nm]'].clip(lower=real['Torque [Nm]'].min())
    df['Tool wear [min]'] = np.clip(df['Tool wear [min]'] + rng.integers(-2, 3, n), 0,
                                    real['Tool wear [min]'].max())

    flags = trigger_flags(df, rng)
    df['TWF'] = ((df['Tool wear [min]'] >= TWF_MIN_WEAR) & (rng.random(n) < q_twf)).astype(int)
    df['HDF'], df['PWF'], df['OSF'] = flags['HDF'], flags['PWF'], flags['OSF']
    df['RNF'] = (rng.random(n) < 0.001).astype(int)
    df['Machine failure'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].max(axis=1)
    return add_features(df)
