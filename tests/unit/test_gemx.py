from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import rimkit.stages.fpa as fpa_stage
from rimkit.exceptions import MotionValidationError
from rimkit.motion import load_source_motion, validate_source_motion
from rimkit.robots.profiles import get_dmr_profile
from rimkit.robots.profiles.fpa import get_fpa_profile


def _write_gemx(path: Path, *, frames: int = 4) -> None:
    torch = pytest.importorskip("torch")
    logits = torch.full((1, frames, 6), -4.0)
    logits[:, :, 1] = 4.0
    logits[:, :, 3] = 4.0
    torch.save(
        {
            "body_params_global": {
                "body_pose": torch.zeros((frames, 228)),
                "global_orient": torch.zeros((frames, 3)),
                "transl": torch.zeros((frames, 3)),
            },
            "net_outputs": {"static_conf_logits": logits},
        },
        path,
    )


def test_gemx_pt_loads_to_floor_normalized_soma77(tmp_path: Path) -> None:
    path = tmp_path / "motion.pt"
    _write_gemx(path)

    loaded = load_source_motion(path, fps_override=60.0)
    schedule = loaded.build_contact_schedule()

    assert loaded.summary.provider == "gem-x"
    assert loaded.summary.container_format == "pt"
    assert loaded.summary.frame_count == 4
    assert loaded.summary.fps == 60.0
    assert loaded.motion.posed_joints.shape == (4, 77, 3)
    assert loaded.motion.global_rot_mats.shape == (4, 77, 3, 3)
    assert loaded.motion.foot_contacts is not None
    assert loaded.motion.foot_contacts.shape == (4, 6)
    assert schedule.contact_source.startswith("gemx_fused_toebase_contacts+time_varying_floor")
    assert np.all(schedule.left_contact_label)
    assert np.all(schedule.right_contact_label)
    assert loaded.gemx_contacts is not None
    assert np.allclose(loaded.gemx_contacts.floor_z, loaded.gemx_contacts.floor_z[0])


def test_gemx_validation_uses_default_30_hz(tmp_path: Path) -> None:
    path = tmp_path / "motion.PT"
    _write_gemx(path)

    summary = validate_source_motion(path)

    assert summary.fps == 30.0
    assert any("30 Hz" in warning for warning in summary.warnings)


def test_gemx_fallback_body_parameters_accept_singleton_batch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    frames = 4
    path = tmp_path / "fallback.pt"
    logits = torch.full((1, frames, 6), -4.0)
    logits[:, :, 1] = 4.0
    logits[:, :, 3] = 4.0
    torch.save(
        {
            "net_outputs": {
                "pred_body_params_global": {
                    "body_pose": torch.zeros((1, frames, 228)),
                    "global_orient": torch.zeros((1, frames, 3)),
                    "transl": torch.zeros((1, frames, 3)),
                },
                "static_conf_logits": logits,
            }
        },
        path,
    )

    loaded = load_source_motion(path)

    assert loaded.summary.frame_count == frames
    assert loaded.motion.posed_joints.shape == (frames, 77, 3)


def test_gemx_rejects_missing_static_contacts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "motion.pt"
    torch.save(
        {
            "body_params_global": {
                "body_pose": torch.zeros((3, 228)),
                "global_orient": torch.zeros((3, 3)),
                "transl": torch.zeros((3, 3)),
            }
        },
        path,
    )

    with pytest.raises(MotionValidationError, match="static_conf_logits"):
        validate_source_motion(path)


def test_gemx_rejects_nonfinite_or_malformed_body_parameters(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "motion.pt"
    _write_gemx(path, frames=3)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["body_params_global"]["body_pose"] = torch.zeros((3, 227))
    torch.save(payload, path)

    with pytest.raises(MotionValidationError, match="body_pose"):
        load_source_motion(path)


def test_source_dispatch_rejects_unknown_container(tmp_path: Path) -> None:
    path = tmp_path / "motion.pkl"
    path.write_bytes(b"not a supported motion")

    with pytest.raises(MotionValidationError, match="expected .npz or .pt"):
        validate_source_motion(path)


@pytest.mark.parametrize(
    "robot_id",
    ("g1", "h2", "r1", "apollo", "oli", "n1", "adam", "t1", "pm01"),
)
def test_gemx_g1_family_profile_overlay(robot_id: str) -> None:
    dmr = get_dmr_profile(robot_id, source_provider="gem-x")
    fpa = get_fpa_profile(robot_id, source_provider="gem-x")

    assert dmr.orientation_smoothing_mode == "quaternion_continuous"
    assert dmr.dmr_initial_nullspace_gain == 1.0
    assert dmr.left_ankle_orientation_joi_key == "lsole"
    assert dmr.right_ankle_orientation_joi_key == "rsole"
    assert dmr.ankle_contact_flatten_strength == 1.0
    assert dmr.ankle_contact_flatten_smooth_time == 0.06
    assert fpa.sole_clearance_quantile == 0.05
    assert fpa.toe_transition_max_step == 0.02
    assert fpa.use_profile_ankle_orientation
    assert fpa.contact_ankle_orientation_weight == 0.15
    assert fpa.joint_correction_smooth_time == 0.10
    assert np.isclose(fpa.joint_correction_max_delta, np.deg2rad(2.0))
    assert fpa.post_ground_root_lower_support_mode == "any"
    assert fpa.post_ground_dual_support_lower_max == 0.03
    assert fpa.post_ground_dual_support_lower_speed == 0.20


def test_gemx_k1_and_h1_overlays_do_not_mutate_kimodo_profiles() -> None:
    k1_default = get_dmr_profile("k1")
    k1_gemx = get_dmr_profile("k1", source_provider="gem-x")
    h1_gemx = get_dmr_profile("h1", source_provider="gem-x")
    h1_fpa = get_fpa_profile("h1", source_provider="gem-x")

    assert k1_default.orientation_smoothing_mode == "rotvec_legacy"
    assert k1_default.dmr_initial_nullspace_gain == 0.0
    assert k1_gemx.orientation_smoothing_mode == "quaternion_continuous"
    assert k1_gemx.dmr_initial_nullspace_gain == 1.0
    assert k1_gemx.ankle_contact_flatten_strength == 0.0
    assert h1_gemx.ankle_contact_flatten_strength == 1.0
    assert h1_gemx.ankle_contact_flatten_smooth_time == 0.10
    assert h1_fpa.use_profile_ankle_orientation
    assert h1_fpa.contact_ankle_orientation_weight == 0.06


def test_run_fpa_forwards_gemx_profile_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seconds = np.asarray((0.0, 1.0 / 30.0), dtype=np.float64)
    targets = SimpleNamespace(robot_id="g1", fps=30.0, seconds=seconds)
    ik = SimpleNamespace(robot_id="g1", fps=30.0, seconds=seconds)
    observed: dict[str, dict[str, Any]] = {}

    def fake_build(*_args: object, **kwargs: Any) -> Any:
        observed["build"] = kwargs
        return targets

    def fake_solve(*_args: object, **kwargs: Any) -> Any:
        observed["solve"] = kwargs
        return ik

    monkeypatch.setattr(fpa_stage, "build_fpa_targets", fake_build)
    monkeypatch.setattr(fpa_stage, "solve_fpa", fake_solve)
    left = np.tile(np.eye(3), (2, 1, 1))
    right = np.tile(np.eye(3), (2, 1, 1))

    result = fpa_stage.run_fpa(
        np.zeros((2, 36)),
        cast(Any, object()),
        cast(Any, object()),
        cast(Any, object()),
        robot_id="g1",
        fps=30.0,
        source_provider="gem-x",
        left_ankle_target_rotations=left,
        right_ankle_target_rotations=right,
    )

    assert result.targets is targets
    assert result.ik is ik
    assert observed["build"]["source_provider"] == "gem-x"
    assert observed["solve"]["source_provider"] == "gem-x"
    assert observed["solve"]["left_ankle_target_rotations"] is left
    assert observed["solve"]["right_ankle_target_rotations"] is right
