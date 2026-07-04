"""Contract tests for the cuRobo 0.8.0 CuroboPlanner rewrite.

These pin the *public contract* of CuroboPlanner (the migration's core invariant:
robot.py and all callers stay unchanged) at the seam of its public methods:
plan_path / plan_batch / plan_grippers, plus the YAML converter and env knobs.

Run in the overlay venv that has cuRobo 0.8.0 installed:
    <scratchpad>/venv-curobo080/bin/python -m pytest \
        external/robotwin_local/envs/robot/test_curobo_planner_contract.py -v

Requires a CUDA GPU. The heavy planner (two warmed-up solver instances) is built
ONCE per module. A real Tron2 validing YAML is copied to a temp dir and converted
there, so no repo file is modified.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pytest
import torch
import transforms3d as t3d
import yaml

# --- make envs.* importable and asset-relative paths resolve -----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROBOTWIN = os.path.abspath(os.path.join(_HERE, "..", ".."))       # external/robotwin_local
_REPO = os.path.abspath(os.path.join(_ROBOTWIN, "..", ".."))       # tron2-in-robotwin
if _ROBOTWIN not in sys.path:
    sys.path.insert(0, _ROBOTWIN)
os.chdir(_ROBOTWIN)  # envs/__init__ loads ./assets/... via relative paths
# Keep contract tests about the planning API rather than scene/table collision.
os.environ.setdefault("CUROBO_NO_TABLE", "1")

_VALIDING = os.path.join(_REPO, "embodiments", "tron2_v5_DACH_validing")
_LEFT_YML = os.path.join(_VALIDING, "curobo_left.yml")
_CONVERTER = os.path.join(_REPO, "tools", "convert_curobo_yaml_080.py")


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_curobo_yaml_080", _CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubPose:
    """Minimal sapien.Pose stand-in: exposes .p (xyz) and .q (wxyz)."""

    def __init__(self, p, q):
        self.p = np.array(p, dtype=float)
        self.q = np.array(q, dtype=float)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (planner is expensive: two warmed-up solvers).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def converted_yml(tmp_path_factory):
    """A real Tron2 validing YAML copied to temp and converted to 0.8.0 schema."""
    d = tmp_path_factory.mktemp("curobo_yml")
    dst = os.path.join(str(d), "curobo_left.yml")
    shutil.copyfile(_LEFT_YML, dst)
    conv = _load_converter()
    assert conv.convert_file(dst) is True  # was old schema -> converted
    assert conv.convert_file(dst) is False  # idempotent
    return dst


@pytest.fixture(scope="module")
def planner_bundle(converted_yml):
    from envs.robot.planner import CuroboPlanner

    with open(converted_yml) as f:
        data = yaml.safe_load(f)
    jn = data["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    retract = data["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"]
    planner = CuroboPlanner(StubPose([0, 0, 0], [1, 0, 0, 0]), jn, list(jn), yml_path=converted_yml)
    return planner, list(jn), list(retract)


@pytest.fixture(scope="module")
def ee_start(planner_bundle):
    """FK of the retract config -> (ee_pos xyz, ee_quat wxyz) in base frame."""
    from curobo.types import JointState

    planner, jn, retract = planner_bundle
    js = JointState.from_position(
        torch.tensor(retract, dtype=torch.float32).cuda().reshape(1, -1), joint_names=jn
    )
    tp = planner.motion_gen.compute_kinematics(js).tool_poses
    if isinstance(tp, dict):
        tp = tp[planner.tool_frame]
    return tp.position.reshape(3).cpu().numpy(), tp.quaternion.reshape(4).cpu().numpy()


def _ee_of(planner, jn, qrow):
    from curobo.types import JointState

    js = JointState.from_position(
        torch.tensor(qrow, dtype=torch.float32).cuda().reshape(1, -1), joint_names=jn
    )
    tp = planner.motion_gen.compute_kinematics(js).tool_poses
    if isinstance(tp, dict):
        tp = tp[planner.tool_frame]
    return tp.position.reshape(3).cpu().numpy(), tp.quaternion.reshape(4).cpu().numpy()


# ===========================================================================
# 1. Single-target success path + trajectory shape / duration sanity
# ===========================================================================
def test_single_success_shape(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    goal = StubPose([p[0], p[1], p[2] + 0.05], q)
    res = planner.plan_path(retract, goal)

    assert res["status"] == "Success"
    assert set(res.keys()) == {"status", "position", "velocity"}
    pos, vel = res["position"], res["velocity"]
    assert pos.ndim == 2 and vel.ndim == 2
    # column count == number of active joints (no all-DOF leak of locked gripper joints)
    assert pos.shape[1] == len(planner.active_joints_name) == 7
    assert vel.shape == pos.shape
    # a real motion of a sane length (not the raw 5000-step buffer, not empty)
    assert 1 < pos.shape[0] < 5000
    # start waypoint matches the commanded start configuration
    assert np.allclose(pos[0], [round(a, 5) for a in retract], atol=1e-3)


# ===========================================================================
# 2. Trajectory truncation: steps == interpolated_last_tstep, columns selected,
#    padding region is terminal repeat (white-box against the raw 0.8.0 result).
# ===========================================================================
def test_trajectory_truncation_matches_last_tstep(planner_bundle, ee_start):
    from curobo.types import JointState

    planner, jn, retract = planner_bundle
    p, q = ee_start
    goal = StubPose([p[0] + 0.10, p[1], p[2] - 0.05], q)

    # Reconstruct plan_path's goal exactly, then inspect the raw (untruncated) result.
    base_target = planner._target_pose_in_base(goal, None)
    raw_goal = planner._make_goal(torch.tensor([base_target], dtype=torch.float32).cuda())
    start = JointState.from_position(
        torch.tensor(planner._current_joint_angles(retract), dtype=torch.float32).cuda().reshape(1, -1),
        joint_names=planner.active_joints_name,
    )
    raw = planner.motion_gen.plan_pose(raw_goal, start, max_attempts=planner.max_attempts)
    assert raw is not None and bool(raw.success.any().item())
    seed = int(torch.argmax(raw.success[0].to(torch.int32)).item())
    last = int(raw.interpolated_last_tstep.view(1, -1)[0, seed].item())
    full7 = raw.interpolated_trajectory.reorder(planner.active_joints_name).position[0, seed].cpu().numpy()
    assert full7.shape[0] >= 5000 - 1  # raw buffer is the full (untruncated) horizon
    # padding region beyond last_tstep is the terminal configuration repeated
    assert np.allclose(full7[last:last + 50], full7[last - 1], atol=1e-5)

    # plan_path (deterministic) must return exactly the truncated + column-selected traj
    res = planner.plan_path(retract, goal)
    assert res["status"] == "Success"
    assert res["position"].shape == (last, 7)
    assert np.allclose(res["position"], full7[:last], atol=1e-5)


# ===========================================================================
# 3. interpolation_dt = 1/250 is in force: velocity ~= position diff / dt
# ===========================================================================
def test_interpolation_dt_velocity_consistency(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    goal = StubPose([p[0] + 0.08, p[1], p[2] - 0.04], q)
    res = planner.plan_path(retract, goal)
    assert res["status"] == "Success"
    pos, vel = res["position"], res["velocity"]
    dt = 1.0 / 250
    fd = (pos[1:] - pos[:-1]) / dt
    err = np.abs(fd - vel[1:])
    assert np.median(err) < 1e-2
    assert abs(planner.interpolation_dt - dt) < 1e-9


# ===========================================================================
# 4. Single-target failure path: unreachable -> {"status": "Fail"} only
# ===========================================================================
def test_single_failure_status_only(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    goal = StubPose([p[0] + 5.0, p[1], p[2]], q)  # far unreachable
    res = planner.plan_path(retract, goal)
    assert res["status"] == "Fail"
    assert set(res.keys()) == {"status"}


# ===========================================================================
# 5. Batch per-pose semantics (anti-goalset guard): mixed reachable /
#    unreachable -> per-pose success tensor, position-wise correspondence.
# ===========================================================================
def test_batch_mixed_reachability_per_pose(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    from envs._GLOBAL_CONFIGS import ROTATE_NUM

    targets = []
    for i in range(ROTATE_NUM):
        if i % 2 == 0:
            targets.append(StubPose([p[0], p[1], p[2] + 0.01 * ((i // 2) + 1)], q))  # reachable
        else:
            targets.append(StubPose([p[0] + 5.0, p[1], p[2]], q))  # unreachable

    res = planner.plan_batch(retract, targets)
    status = res["status"]
    assert len(status) == ROTATE_NUM
    # NOT goalset ("reach any one" -> all True) and NOT total failure
    assert not all(s == "Success" for s in status)
    assert not all(s == "Failure" for s in status)
    # per-pose correspondence: every unreachable index fails, every reachable succeeds
    for i in range(ROTATE_NUM):
        expected = "Success" if i % 2 == 0 else "Failure"
        assert status[i] == expected, f"pose {i}: {status[i]} != {expected}"


def test_batch_position_shape_and_columns(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    from envs._GLOBAL_CONFIGS import ROTATE_NUM

    targets = [StubPose([p[0], p[1], p[2] + 0.01 * (i + 1)], q) for i in range(ROTATE_NUM)]
    res = planner.plan_batch(retract, targets)
    assert res["position"].shape[0] == ROTATE_NUM
    assert res["position"].shape[2] == len(planner.active_joints_name) == 7
    assert res["velocity"].shape == res["position"].shape
    # per-pose trajectory is indexable and has a length (choose_best_pose contract)
    assert len(res["position"][0]) == res["position"].shape[1]


# ===========================================================================
# 6. Constraint plan: partial-pose hold takes effect AND resets cleanly
#    (no leak into the subsequent unconstrained plan).
# ===========================================================================
def test_constraint_applies_and_resets(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    # goal requires a 40-degree EE reorientation so a hold-orientation constraint bites
    R = t3d.axangles.axangle2mat([0, 0, 1], np.deg2rad(40))
    goal_q = t3d.quaternions.qmult(t3d.quaternions.mat2quat(R), q)
    goal = StubPose([p[0], p[1], p[2]], goal_q)

    base = planner.plan_path(retract, goal)
    con = planner.plan_path(retract, goal, constraint_pose=[1, 1, 1, 0, 0, 0])  # hold orientation
    after = planner.plan_path(retract, goal)

    assert base["status"] == "Success"
    assert con["status"] == "Success"
    assert after["status"] == "Success"

    def maxdiff(a, b):
        n = min(len(a), len(b))
        return float(np.abs(a[:n] - b[:n]).max())

    # constraint measurably changed the plan
    assert maxdiff(base["position"], con["position"]) > 1e-3
    # reset was clean: the unconstrained plan after the constrained one == baseline
    assert maxdiff(base["position"], after["position"]) < 1e-4


# ===========================================================================
# 7. No state leak across sequential plans (determinism + no cross-plan bleed)
# ===========================================================================
def test_no_state_leak_sequential(planner_bundle, ee_start):
    planner, jn, retract = planner_bundle
    p, q = ee_start
    g1 = StubPose([p[0] + 0.06, p[1], p[2] - 0.02], q)
    g2 = StubPose([p[0], p[1] + 0.05, p[2] + 0.03], q)

    a1 = planner.plan_path(retract, g1)
    b = planner.plan_path(retract, g2)  # a different plan in between
    a2 = planner.plan_path(retract, g1)
    assert a1["status"] == a2["status"] == b["status"] == "Success"
    # planning g1 again after an intervening g2 gives the identical trajectory
    assert np.allclose(a1["position"], a2["position"], atol=1e-6)

    # a batch plan in between does not perturb the single-target result either
    from envs._GLOBAL_CONFIGS import ROTATE_NUM

    _ = planner.plan_batch(retract, [StubPose([p[0], p[1], p[2] + 0.01 * (i + 1)], q) for i in range(ROTATE_NUM)])
    a3 = planner.plan_path(retract, g1)
    assert np.allclose(a1["position"], a3["position"], atol=1e-6)


# ===========================================================================
# 8. Fail-loud on un-converted (0.7.x) schema.
# ===========================================================================
def test_fail_loud_on_old_schema():
    from envs.robot.planner import CuroboPlanner

    with pytest.raises(RuntimeError, match=r"convert_curobo_yaml_080"):
        # The repo's on-disk YAML is still 0.7.x schema (ee_link / retract_config).
        # _build_robot_dict raises before any GPU work.
        CuroboPlanner(StubPose([0, 0, 0], [1, 0, 0, 0]), ["j"], ["j"], yml_path=_LEFT_YML)


# ===========================================================================
# 9. YAML converter: surgical, schema-only, path-preserving, idempotent.
# ===========================================================================
def test_yaml_converter_surgical_and_idempotent():
    conv = _load_converter()
    with open(_LEFT_YML) as f:
        original = f.read()
    converted = conv.convert_text(original)

    # schema keys flipped
    assert "tool_frames:" in converted and "ee_link:" not in converted
    assert "default_joint_position:" in converted and "retract_config:" not in converted
    assert "use_usd_kinematics:" not in converted
    # asset_root_path: null removed
    assert not any(l.strip().startswith("asset_root_path:") for l in converted.splitlines())

    # path values preserved byte-for-byte
    for line in original.splitlines():
        if "urdf_path:" in line or "collision_spheres:" in line:
            assert line in converted.splitlines()
    # comments preserved
    assert original.splitlines()[0] in converted.splitlines()
    # tool_frames carries the exact ee_link value
    assert 'tool_frames: ["tcp_L_Link"]' in converted

    # idempotent
    assert conv.convert_text(converted) == converted
    assert conv.is_already_converted(converted) is True


def test_yaml_converter_preserves_inline_comment_on_ee_link():
    conv = _load_converter()
    src = (
        "robot_cfg:\n"
        "  kinematics:\n"
        '    ee_link: "tcp_L_Link"  # end effector\n'
        "    cspace:\n"
        "      retract_config: [0.0, 1.0]  # home\n"
    )
    out = conv.convert_text(src)
    # comment kept OUTSIDE the flow list -> still valid YAML that round-trips
    assert 'tool_frames: ["tcp_L_Link"]  # end effector' in out
    assert "default_joint_position: [0.0, 1.0]  # home" in out
    parsed = yaml.safe_load(out)
    assert parsed["robot_cfg"]["kinematics"]["tool_frames"] == ["tcp_L_Link"]
    assert parsed["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"] == [0.0, 1.0]


# ===========================================================================
# 10. Env knobs.
# ===========================================================================
def test_knob_trajopt_seeds_flows_to_solver(planner_bundle, monkeypatch):
    from envs.robot.planner import CuroboPlanner

    planner, _, _ = planner_bundle
    # default: 1 seed, flowed into the built trajopt solver
    assert planner.trajopt_seeds == 1
    assert planner.motion_gen.trajopt_solver.config.num_seeds == planner.trajopt_seeds
    # env mapping TRON2_CUROBO_TRAJOPT_SEEDS -> num_trajopt_seeds
    monkeypatch.setenv("TRON2_CUROBO_TRAJOPT_SEEDS", "4")
    assert CuroboPlanner._read_knobs()["trajopt_seeds"] == 4


def test_knob_max_attempts_passed_to_plan_pose(planner_bundle, ee_start, monkeypatch):
    from envs.robot.planner import CuroboPlanner

    planner, jn, retract = planner_bundle
    p, q = ee_start
    assert planner.max_attempts == 10  # default

    captured = {}
    orig = planner.motion_gen.plan_pose

    def spy(goal, start, max_attempts=5, **kw):
        captured["max_attempts"] = max_attempts
        return None  # force a Fail return, exercising the failure branch too

    monkeypatch.setattr(planner.motion_gen, "plan_pose", spy)
    res = planner.plan_path(retract, StubPose([p[0], p[1], p[2] + 0.05], q))
    assert captured["max_attempts"] == planner.max_attempts
    assert res["status"] == "Fail"

    # env mapping TRON2_CUROBO_MAX_ATTEMPTS -> max_attempts
    monkeypatch.setenv("TRON2_CUROBO_MAX_ATTEMPTS", "7")
    assert CuroboPlanner._read_knobs()["max_attempts"] == 7


def test_knob_graph_seeds_deprecation_warning(monkeypatch):
    from envs.robot.planner import CuroboPlanner

    monkeypatch.setenv("TRON2_CUROBO_GRAPH_SEEDS", "3")
    with pytest.warns(DeprecationWarning, match="TRON2_CUROBO_GRAPH_SEEDS"):
        knobs = CuroboPlanner._read_knobs()
    assert knobs["graph_seeds"] == 3

    # unset -> no warning
    monkeypatch.delenv("TRON2_CUROBO_GRAPH_SEEDS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CuroboPlanner._read_knobs()
