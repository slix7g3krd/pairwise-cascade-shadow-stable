"""题目所述六个级联阴影稳定性修复的行为回归测试。"""

import math

import pytest

from graphics import cascade_shadow_stable as cs


def test_texel_snap_uses_light_space_grid_and_preserves_depth():
    snapped = cs.texel_snap([1.23, -2.34, 7.0], 0.5)
    assert snapped == [1.0, -2.5, 7.0]

    report = cs.texel_snap(
        {"min": [-0.7, -0.2], "max": [0.8, 1.3], "texel_size": 0.5}
    )
    assert report["min"] == [-1.0, -0.5]
    assert report["max"] == [1.0, 1.5]


def test_discrete_hysteresis_matches_deadband_contract():
    assert int(cs.switch_hysteresis(1, 2, 2)) == 1
    assert int(cs.switch_hysteresis(1, 3, 2)) == 3


def test_distance_hysteresis_creates_no_switch_band():
    splits = [10.0, 20.0, 40.0]

    waiting = cs.switch_hysteresis(
        distance=10.4, splits=splits, current=0, ratio=0.1
    )
    assert int(waiting) == 0
    assert waiting["switched"] is False

    outward = cs.switch_hysteresis(
        distance=11.1, splits=splits, current=0, ratio=0.1
    )
    assert int(outward) == 1
    assert outward["switched"] is True

    inward_wait = cs.switch_hysteresis(
        distance=9.6, splits=splits, current=1, ratio=0.1
    )
    assert int(inward_wait) == 1

    inward = cs.switch_hysteresis(
        distance=8.9, splits=splits, current=1, ratio=0.1
    )
    assert int(inward) == 0


def test_same_frame_capture_pairs_matrices_and_detects_stale_sources():
    camera = [[float(i + j) for j in range(4)] for i in range(4)]
    packet = cs.same_frame_matrix(12, camera, None)

    assert packet["frame"] == 12
    assert packet["camera_frame"] == packet["shadow_frame"] == 12
    assert packet["lag_frames"] == 0
    assert packet["aligned"] is True

    stale = cs.same_frame_matrix(
        {
            "camera_snapshot": {"frame": 11, "matrix": cs.identity_matrix()},
            "shadow_snapshot": {"frame": 12, "matrix": cs.identity_matrix()},
            "strict": True,
        }
    )
    assert stale["source_aligned"] is False
    assert stale["source_lag_frames"] == 1
    assert stale["ok"] is False


def test_filter_radius_uses_each_cascades_texel_size():
    result = cs.filter_per_cascade(texel_sizes=[0.01, 0.04, 0.16])
    assert result["radii"] == [0.01, 0.04, 0.16]
    assert result["radii"][-1] > result["radii"][0]

    configured = cs.filter_per_cascade(
        resolutions=[128, 64], world_sizes=[10.0, 20.0]
    )
    assert configured["texel_sizes"] == [10.0 / 128.0, 20.0 / 64.0]
    assert configured["radii"][1] == configured["texel_sizes"][1]


def test_fov_coverage_grows_for_wide_angle():
    narrow = cs.fov_coverage(45.0, 1.0, 0.1, 50.0)
    wide = cs.fov_coverage(90.0, 1.0, 0.1, 50.0)

    assert narrow["fov_degrees"] == pytest.approx(45.0)
    assert wide["width"] > narrow["width"]
    assert wide["coverage_radius"] > narrow["coverage_radius"]

    radian = cs.fov_coverage(math.pi / 2.0, 1.0, 0.1, 1.0, unit="rad")
    assert radian["far_plane"]["height"] == pytest.approx(2.0)


def test_seam_weight_and_blend_are_continuous():
    assert cs.seam_weight(9.0, 10.0, 2.0) == 0.0
    assert cs.seam_weight(10.0, 10.0, 2.0) == pytest.approx(0.5)
    assert cs.seam_weight(11.0, 10.0, 2.0) == 1.0

    value = cs.seam_blend(0.2, 0.4, 0.5)
    assert value == pytest.approx(0.3)


def test_fit_produces_aligned_valid_cascades():
    result = cs.fit(cascades=3, fov=70.0, aspect=1.7, resolution=512)
    assert result["ok"] is True
    assert len(result["cascades"]) == 3
    assert result["validation"]["errors"] == []
    texels = [c["texel_size"] for c in result["cascades"]]
    assert texels[1] > texels[0]
    assert texels[2] > texels[1]


def test_support_contracts_have_stable_reports():
    assert cs.determinism(11)["values"] == cs.determinism(11)["values"]
    assert cs.validation(1)["ok"] is True
    assert cs.audit(1)["event_count"] == 1
    assert cs.fallback(1)["mode"].startswith("uniform")
    assert cs.metrics(1.0)["rms"] == 1.0
    assert cs.cascade_count([10, 20, 40]) == 3
    assert cs.cascade_count(1) == 1
