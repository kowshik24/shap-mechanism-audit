import numpy as np
from pdm_audit.exact_oracle import ALIGNED, exact_shap, statistics
from pdm_audit.conditional_audit import (sample_batches, observed_overlap, overlap_bins,
    batch_scores, coupled_context_counts, permute_context_counts, residual)


def test_vector_scores_match_scalar_and_observed_overlap_uses_counts():
    rng = np.random.default_rng(20); counts, u = sample_batches(1000, rng, 8)
    score, _, valid = batch_scores(.1+.8*ALIGNED, counts)
    for k, row in enumerate(counts):
        scalar = statistics(exact_shap(.1+.8*ALIGNED, row/row.sum()), row)
        assert np.isclose(score[k], max(x['S'] for x in scalar.values()))
    assert valid.all() and np.all(counts.sum(axis=1) == 1000)
    assert not np.allclose(observed_overlap(counts), u)
    assert np.all((overlap_bins(counts) >= 0) & (overlap_bins(counts) < 3))


def test_strength_and_permutation_preserve_true_mechanisms_and_counts():
    rng = np.random.default_rng(21); counts, _ = sample_batches(1000, rng, 8)
    alternatives = coupled_context_counts(counts, [0., .01, .5], rng)
    for rows in alternatives:
        assert np.array_equal(rows.reshape(8, 8, 2).sum(axis=2), counts.reshape(8, 8, 2).sum(axis=2))
        permuted = permute_context_counts(rows, rng)
        assert np.array_equal(permuted[:, 1::2].sum(axis=1), rows[:, 1::2].sum(axis=1))
        assert np.array_equal(permuted.reshape(8, 8, 2).sum(axis=2), rows.reshape(8, 8, 2).sum(axis=2))
    a = np.arange(8)/10
    # Constant reference offsets cancel under target-size tuning.
    assert np.allclose(residual(a, np.full(8, .2), np.ones(8, bool))+.2, a)
