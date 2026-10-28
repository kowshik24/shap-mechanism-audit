"""Review-specific diagnostics without changing the historical audit definition."""
from itertools import combinations
import numpy as np
from .audit import holm, permutation_p
from .metrics import clopper_pearson


def group_statistics(attribution, own_rows, other_rows, expected):
    """Shares of group-mean absolute attribution for an explicit comparison cohort."""
    own = attribution.loc[own_rows].mean()
    other = attribution.loc[other_rows].mean()
    expected = [f for f in expected if f in own.index]
    a, b = own / own.sum(), other / other.sum()
    excess = (a-b).drop(expected)
    return {'R': float(a[expected].sum()), 'D': float(a[expected].sum()-b[expected].sum()),
            'S': float(excess.max()), 'S_feature': str(excess.idxmax()),
            'n_own': len(own_rows), 'n_other': len(other_rows)}


def monte_carlo_family(observed, draws, alpha=0.05):
    """Simultaneous binomial tail bounds, propagated through monotone Holm adjustment.

    The interval target is the limiting reference exceedance probability, not the
    finite plus-one estimate. Coverage is conditional on the observed statistic and
    iid reference draws. It does not validate a reference-design hypothesis.
    """
    count = len(observed)
    p, low, high, details = {}, {}, {}, {}
    for key, value in observed.items():
        null = np.asarray(draws[key]); hits = int((null >= value).sum())
        p[key] = permutation_p(value, null)
        low[key], high[key] = clopper_pearson(hits, len(null), alpha / count)
        details[key] = {'hits': hits, 'K': len(null), 'p': p[key],
                        'tail_interval': [low[key], high[key]],
                        'reference_mean': float(null.mean()),
                        'difference': float(value-null.mean())}
    adj, lo, hi = holm(p), holm(low), holm(high)
    for key in details:
        details[key].update(p_holm=adj[key], holm_interval=[lo[key], hi[key]],
                            mc_decision_stable=bool(hi[key] < .05 or lo[key] >= .05))
    return details


def disjoint_feature_decoys(attribution, rows, expected):
    """Enumerate equal-cardinality sets disjoint from the expected feature set.

    This preserves all predictions and attribution values. It is a specificity
    diagnostic and does not represent an intervention on the data-generating process.
    """
    weights = attribution.loc[rows].mean(); shares = weights / weights.sum()
    expected = [x for x in expected if x in shares.index]
    pool = [x for x in shares.index if x not in expected]
    sets = list(combinations(pool, len(expected)))
    values = np.array([shares[list(s)].sum() for s in sets], dtype=float)
    obs = float(shares[expected].sum())
    return {'R': obs, 'set_size': len(expected), 'n_decoys': len(values),
            'decoy_mean': float(values.mean()), 'decoy_max': float(values.max()),
            'fraction_decoys_ge_observed': float((values >= obs).mean()),
            'observed_minus_decoy_mean': float(obs-values.mean())}
