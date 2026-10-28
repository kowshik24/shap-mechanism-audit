import numpy as np
from pdm_audit.exact_oracle import exact_shap, STATES, STATE_WEIGHTS
from pdm_audit.reference_transfer import scores
from pdm_audit.score_design import contributions, excess


def test_frozen_and_batch_shap_match_exact_enumeration():
    rng = np.random.default_rng(79)
    values = rng.random(16)
    counts = rng.integers(1, 100, (3, 16))
    frozen = rng.dirichlet(np.ones(16))
    for background in [None, frozen]:
        actual = contributions(values, counts, background)
        for i, row in enumerate(counts):
            law = row / row.sum() if background is None else background
            assert np.allclose(actual[i], abs(exact_shap(values, law)), atol=1e-13)
    a, l, v = excess(contributions(values, counts), counts)
    aa, ll, vv = scores(values, counts)
    assert np.allclose(a, aa) and np.allclose(l, ll) and np.array_equal(v, vv)


def test_healthy_comparison_uses_healthy_rows_and_map_bounds_cover_completions():
    values = (STATES[:, 3] | STATES[:, 1] | STATES[:, 2]).astype(float)
    counts = np.ones((1, 16))
    local = contributions(values, counts)
    _, healthy, _ = excess(local, counts, normalize=False, healthy=True)
    _, failure, _ = excess(local, counts, normalize=False)
    # A cohort change can change localization; do not assert universal improvement.
    assert not np.array_equal(healthy, failure)
    upper, lower, valid = excess(local, counts, bounds=True)
    assert valid.all() and (lower <= upper).all() and np.isfinite(lower).all()
