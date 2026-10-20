"""
AI4I 2020 generated afresh at arbitrary size, following the generating process documented with the
dataset [Matzka 2020], with no row copied from the release:

  Type              L / M / H with probability 0.6 / 0.3 / 0.1
  air temperature   N(300, 2) K
  process temp.     air temperature + a difference drawn N(10, 1) K, with correlation -0.70 between
                    air temperature and the difference, as in the release. The documented process
                    uses random walks; we draw rows independently instead, because the model never
                    sees row order and a slowly varying walk makes the HDF rate of a sample of
                    10^4 rows depend on a handful of excursions (0-1.4% across seeds).
  torque            N(40, 10) Nm, redrawn below the release minimum of 3.8 Nm
  rotational speed  quadratic in 1/torque fitted to the release, plus Gaussian noise with the
                    residual s.d. (the release describes speed as derived from power plus noise)
  tool wear         grows by 5 / 3 / 2 min per row for H / M / L; each tool has a life drawn from
                    U(200, 240) min, at which it either fails (TWF) or is replaced, then wear resets

Flags follow the published trigger rules for HDF, PWF and OSF (which reproduce the released flags
exactly), TWF from the tool-life process with the failure probability calibrated to the released TWF
rate, and RNF with probability 0.001. Machine failure is exactly the OR of the five flags.
"""
import numpy as np
import pandas as pd

from .data import add_features, load
from .config import STRAIN_LIMIT

TYPES, TYPE_P = np.array(['L', 'M', 'H']), [0.6, 0.3, 0.1]
WEAR_STEP = {'L': 2, 'M': 3, 'H': 5}
TORQUE_MIN = 3.8


def _speed_model():
    real = load()
    t, s = real['Torque [Nm]'].values, real['Rotational speed [rpm]'].values
    coef = np.polyfit(1 / t, s, 2)
    return coef, float(np.std(s - np.polyval(coef, 1 / t)))


RHO_AIR_DIFF = -0.70


def _tool_wear(types, rng, q_fail):
    wear, twf = np.empty(len(types)), np.zeros(len(types), dtype=int)
    w, life = 0.0, rng.uniform(200, 240)
    for i, t in enumerate(types):
        w += WEAR_STEP[t]
        wear[i] = w
        if w >= life:
            twf[i] = rng.random() < q_fail
            w, life = 0.0, rng.uniform(200, 240)
    return wear, twf


def simulate(n, seed=0, q_fail=0.40):
    """q_fail: probability that a tool reaching its life fails rather than being replaced
    (0.40 gives the released TWF rate of about 0.46%)."""
    rng = np.random.default_rng(seed)
    coef, speed_sd = _speed_model()
    types = rng.choice(TYPES, n, p=TYPE_P)
    z_air, z2 = rng.normal(size=n), rng.normal(size=n)
    air = 300.0 + 2.0 * z_air
    process = air + 10.0 + RHO_AIR_DIFF * z_air + np.sqrt(1 - RHO_AIR_DIFF ** 2) * z2
    torque = rng.normal(40, 10, n)
    while (low := torque < TORQUE_MIN).any():
        torque[low] = rng.normal(40, 10, low.sum())
    speed = np.polyval(coef, 1 / torque) + rng.normal(0, speed_sd, n)
    wear, twf = _tool_wear(types, rng, q_fail)

    df = pd.DataFrame({'Type': types, 'Air temperature [K]': air, 'Process temperature [K]': process,
                       'Rotational speed [rpm]': speed, 'Torque [Nm]': torque, 'Tool wear [min]': wear})
    power = torque * speed * 2 * np.pi / 60
    df['TWF'] = twf
    df['HDF'] = ((process - air < 8.6) & (speed < 1380)).astype(int)
    df['PWF'] = ((power < 3500) | (power > 9000)).astype(int)
    df['OSF'] = (wear * torque > df['Type'].map(STRAIN_LIMIT)).astype(int)
    df['RNF'] = (rng.random(n) < 0.001).astype(int)
    df['Machine failure'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].max(axis=1)
    return add_features(df)
