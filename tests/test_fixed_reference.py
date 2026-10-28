import numpy as np
from itertools import product
from pdm_audit.exact_oracle import STATES,ALIGNED,exact_shap,statistics
from pdm_audit.fixed_reference import population_probabilities,batch_max_score


def test_overlap_population_has_correct_joint_marginals():
    low,high = np.array([.02,.03,.04]),np.array([.2,.15,.25])
    q,u = .3,.7
    mass = population_probabilities(q,low,high,u)
    p0 = (1-q)*low[0]+q*high[0]
    p1 = (1-q)*(low[1]+(1-low[1])*u*low[0])+q*(high[1]+(1-high[1])*u*high[0])
    assert np.isclose(mass@STATES[:,0],p0)
    assert np.isclose(mass@STATES[:,1],p1)
    assert np.isclose(mass@STATES[:,3],.01+.98*p0)
    # Exact limiting case agrees with enumeration of nested Z0=Z1.
    p = .05
    nested = population_probabilities(0,[p,0,p],[p,0,p],1)
    assert np.isclose(nested[STATES[:,0] != STATES[:,1]].sum(),0)


def test_batch_combines_signed_values_before_absolute_values():
    rng = np.random.default_rng(41)
    counts = rng.integers(1,100,size=(4,16))
    values = .05+.9*ALIGNED
    values = values+.01*rng.normal(size=16)
    batched = batch_max_score(values,counts)
    scalar = [max(s['S'] for s in statistics(exact_shap(values,c/c.sum()),c).values()) for c in counts]
    assert np.allclose(batched,scalar,atol=1e-12)


def test_multinomial_law_matches_explicit_latent_event_enumeration():
    q,u = .23,.6
    low,high = [.025,.03,.01],[.11,.27,.18]
    expected = np.zeros(16)
    for regime,p in [(0,np.array(low)),(1,np.array(high))]:
        for base in product([0,1],repeat=3):
            base = np.array(base)
            mass = (q if regime else 1-q)*np.prod(np.where(base,p,1-p))
            for nested,flip in product([0,1],repeat=2):
                z = base.copy();z[1] |= z[0]*nested
                context = z[0]^flip
                index = int(np.array([*z,context])@np.array([8,4,2,1]))
                expected[index] += mass*(u if nested else 1-u)*(.01 if flip else .99)
    assert np.allclose(population_probabilities(q,low,high,u),expected,atol=1e-15)
