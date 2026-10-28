import itertools
import numpy as np
import pytest
from pdm_audit.partial_map import identification_ranges,conservative_rank_cut


def test_sharp_ranges_equal_every_admissible_map_completion():
    a=np.array([.12,.28,.41,.19]);b=np.array([.31,.14,.22,.33]);delta=a-b
    for states in itertools.product(range(3),repeat=4):
        p=[i for i,x in enumerate(states) if x==0];n=[i for i,x in enumerate(states) if x==1];u=[i for i,x in enumerate(states) if x==2]
        result=identification_ranges(a,b,p,n);values=[]
        for bits in itertools.product([0,1],repeat=len(u)):
            e=p+[j for j,bit in zip(u,bits) if bit];outside=list(set(range(4))-set(e))
            values.append([sum(a[e]),sum(delta[e]),max(delta[outside],default=-np.inf)])
        values=np.asarray(values)
        for j,key in enumerate(['R','D','S']):
            assert np.allclose(result[key],[values[:,j].min(),values[:,j].max()])


def test_interval_rank_warning_implies_every_exact_rank_warning():
    intervals=[[-2,-1],[0,1],[2,3],[4,5]];upper=[x[1] for x in intervals]
    cut,k=conservative_rank_cut(upper,alpha=.4)
    assert k==3 and cut==3
    for actual in itertools.product(*intervals):
        exact=np.sort(actual)[k-1]
        for lower in [-np.inf,2.,4.]:
            for candidate in [lower,lower+1]:
                assert not lower>cut or candidate>exact
    assert conservative_rank_cut([0]*18)[0]==np.inf
    assert conservative_rank_cut([0]*19)==(0,19)


def test_missing_cohort_and_empty_exclusion_do_not_create_a_warning():
    a=[.8,.2];result=identification_ranges(a,None,[0])
    assert result['D'] is None and result['S'] is None and result['R']==[.8,1.]
    result=identification_ranges(a,[.2,.8],[0])
    assert result['S'][0]==-np.inf
    with pytest.raises(ValueError):identification_ranges(a,a,[0],[0])
    with pytest.raises(ValueError):identification_ranges([0,0],a,[])
