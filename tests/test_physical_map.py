import numpy as np
import pytest
from pdm_audit.physical_map import holm_adjust,randomized_response,path_status,map_coverage


def test_missing_graph_paths_do_not_exclude_effects():
    a=np.array([[0,1,0],[0,0,1],[0,0,0]])
    assert path_status(a,['x','y','z'],'x','z')=='path'
    assert path_status(a,['x','y','z'],'z','x')=='no_listed_path'
    result=map_coverage(['x','y','z'],['y','z'],[])
    assert result['unknown']==['x'] and not result['complete_relative_to_universe']
    with pytest.raises(ValueError):map_coverage(['x'],['x'],['x'])


def test_randomization_counts_and_global_holm_family():
    flags=np.tile([0,1],50);values=10*flags.astype(float)
    tests=randomized_response(values,flags)['tests']
    assert tests[0]['counts']==[[50,0],[0,50]] and tests[0]['p']<1e-20
    assert tests[1]['p']==1
    assert np.allclose(holm_adjust([.03,.001,.04,.02]),[.06,.004,.06,.06])
    assert randomized_response(np.ones(100),flags)['tests'][0]['p']==1
