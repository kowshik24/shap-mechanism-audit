import numpy as np
from pdm_audit.exact_oracle import ALIGNED,STATES
from pdm_audit.reference_transfer import scores,shifted_counts


def test_intervention_excess_separates_stable_proxy_from_mechanism_only():
    counts=np.ones((3,16),dtype=int)*100
    reference=ALIGNED.copy()
    proxy=STATES[:,[1,2,3]].max(axis=1).astype(float)
    a,local,valid=scores(reference,counts,'flip')
    assert valid.all() and np.all(a==0) and np.all(local==0)
    a,local,valid=scores(proxy,counts,'flip')
    assert valid.all() and np.all(a>0) and np.all(local>0)
    # Decisions can coincide on the observed duplicate-input support even though
    # the complete response test detects a difference off that support.
    duplicate=STATES[:,0]==STATES[:,3]
    assert np.array_equal(reference[duplicate],proxy[duplicate])
    assert np.any(reference[~duplicate]!=proxy[~duplicate])


def test_decoupling_preserves_mechanisms_and_null_score_but_reverses_context():
    counts=np.full((8,16),100,dtype=int)
    changed=shifted_counts(counts,np.random.default_rng(7))
    baseline=scores(ALIGNED,counts)[0]
    mechanisms=counts.reshape(8,8,2).sum(axis=2)
    for law in changed:
        assert np.array_equal(law.reshape(8,8,2).sum(axis=2),mechanisms)
        assert np.allclose(scores(ALIGNED,law)[0],baseline)
    expected=np.where(STATES[::2,0][None,:]==0,mechanisms,0)
    assert np.array_equal(changed[-1,:,1::2],expected)
