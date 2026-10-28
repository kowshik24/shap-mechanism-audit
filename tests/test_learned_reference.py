import numpy as np
from pdm_audit.exact_oracle import ALIGNED, MISALIGNED
from pdm_audit.learned_reference import functional_gate,audit_data


def test_gate_checks_unobserved_states_and_probability_dependence():
    assert functional_gate(.05+.9*ALIGNED)['eligible']
    assert not functional_gate(MISALIGNED)['eligible']
    # Correct decisions alone do not establish context invariance.
    values = .1+.7*ALIGNED
    values[1::2] += .1
    gate = functional_gate(values)
    assert gate['full_state_decision_agreement'] == 1
    assert not gate['eligible'] and gate['material_context_dependence']
    wrong = .05+.9*ALIGNED; wrong[14] = .1
    assert not functional_gate(wrong)['eligible']


def test_overlap_shift_is_paired_and_has_no_outcome_or_context_shift():
    x,shifted,permuted,_ = audit_data(10000,410000)
    assert np.array_equal(x[:,[0,2,3]],shifted[:,[0,2,3]])
    assert np.all(shifted[:,1] >= x[:,1])
    assert np.array_equal(x[:,:3].max(axis=1),shifted[:,:3].max(axis=1))
    assert np.array_equal(x[:,:3],permuted[:,:3])
    assert np.array_equal(np.sort(x[:,3]),np.sort(permuted[:,3]))
