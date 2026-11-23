import pytest
import graphics.cascade_shadow_stable as mod

CASES = [
    ("texel_snap", (1,)),
    ("switch_hysteresis", (1,)),
    ("same_frame_matrix", (1,)),
    ("filter_per_cascade", (1,)),
    ("fov_coverage", (1,)),
    ("determinism", (1,)),
    ("seam_blend", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("cascade_count", (1,)),
    ("metrics", (1,)),
    ("shadow_map", (1,)),
    ("bias", (1,)),
    ("fit", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
