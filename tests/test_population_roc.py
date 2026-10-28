import numpy as np
from pdm_audit.population_roc import upper_ranks,matched_boundary,boundary_rate,population_pvalues


def test_matched_boundary_handles_atoms_without_claiming_a_deployment_cut():
    null=np.r_[np.full(30,.1),np.full(70,.9)]
    cut=matched_boundary(null,.05)
    assert cut[0]==.1 and np.isclose(cut[1],1/6)
    assert np.isclose(boundary_rate(null,cut),.05)
    assert np.isclose(boundary_rate(np.array([0.,.1,.9]),cut),(1+1/6)/3)


def test_upper_tail_ranks_and_abstention_for_unavailable_cells():
    cal=np.zeros(20);u=np.linspace(.01,.99,20)
    p=upper_ranks(cal,u,np.array([0.,0.,1.]),np.array([0.,1.,.5]),np.ones(3,bool),True)
    assert np.allclose(p,[1.,1/21,1/21])
    strict=upper_ranks(cal,u,np.array([0.,1.]),np.ones(2),np.ones(2,bool),False)
    assert np.allclose(strict,[1.,1/21])
    missing=population_pvalues(cal,np.zeros(20),u,np.array([1.,1.]),np.array([0,1]),
                              np.ones(2),np.ones(2,bool),False,2)
    assert np.allclose(missing,[1/21,2.])
