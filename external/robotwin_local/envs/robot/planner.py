import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import os
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS


try:
    # ********************** CuroboPlanner (cuRobo 0.8.0 API) **********************
    # cuRobo 0.8.0 ("cuRoboV2", Apache-2.0) is a full refactor of 0.7.x: the old
    # curobo.wrap / curobo.types.math / curobo.util import paths are gone and the
    # motion generation entry points changed. This class keeps the *exact same
    # public contract* as the 0.7.8 implementation so envs/robot/robot.py and every
    # task/collection/debug script keep working unchanged:
    #   __init__(robot_origion_pose, active_joints_name, all_joints, yml_path)
    #   plan_path(...)  -> {"status": "Success"/"Fail", "position", "velocity"}
    #   plan_batch(...) -> {"status": np.array(["Success"/"Failure", ...]),
    #                        "position", "velocity"}
    #   plan_grippers(...)
    import copy
    import time
    import warnings

    import torch
    import yaml

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.types import JointState, Pose as CuroboPose, GoalToolPose
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from curobo.logging import setup_logger

    setup_logger(level="error", logger_name="curobo")

    # constraint_pose semantics: 0.7.8 fed constraint_pose straight into
    # PoseCostMetric(hold_partial_pose=True, hold_vec_weight=constraint_pose) as a
    # 6-vector of *running-waypoint* weights in [rot(3), trans(3)] order (see the
    # vendored 0.7.8 curobo pose_cost.py: vec_weight[0:3]=rot, [3:6]=pos).
    #
    # In cuRobo 0.8.0 that PoseCostMetric path is DEAD: solver.update_pose_cost_metric
    # forwards `pose_cost_metric` to the cost managers, but RobotCostManager.update_params
    # only consumes `tool_pose_criteria`/`dt` and silently drops `pose_cost_metric`
    # (verified empirically: holding all 6 DOF changed nothing). The live mechanism is
    # ToolPoseCriteria.non_terminal_pose_axes_weight_factor applied via
    # update_tool_pose_criteria, whose axis order is [trans(3), rot(3)] -- the two
    # halves are swapped vs 0.7.8, so we swap them below.
    #
    # Fallback default criteria (standard cuRobo trajopt default: only the terminal
    # waypoint tracks the full goal, running waypoints are free). Used only if the
    # live default cannot be read at construction time.
    _DEFAULT_TOOL_POSE_CRITERIA = {
        "terminal_pose_axes_weight_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "non_terminal_pose_axes_weight_factor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "terminal_pose_convergence_tolerance": [0.0, 0.0],
        "non_terminal_pose_convergence_tolerance": [0.0, 0.0],
        "project_distance_to_goal": False,
    }

    # Old (0.7.x) kinematics schema keys that 0.8.0 no longer accepts. If a YAML
    # still carries them, it has not been run through the converter and would
    # otherwise fail deep inside cuRobo with an opaque error, so we fail loud.
    _OLD_SCHEMA_HINT = (
        "This curobo YAML still uses the cuRobo 0.7.x schema. Run the converter:\n"
        "    python tools/convert_curobo_yaml_080.py <yml_path>\n"
        "(ee_link -> tool_frames, cspace.retract_config -> cspace.default_joint_position)."
    )

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            setup_logger(level="error", logger_name="curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints
            knobs = self._read_knobs()
            self.trajopt_seeds = knobs["trajopt_seeds"]
            self.graph_seeds = knobs["graph_seeds"]
            self.max_attempts = knobs["max_attempts"]
            self.interpolation_dt = knobs["interpolation_dt"]

            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            self._robot_dict = self._build_robot_dict(yml_data)
            self.tool_frame = self._robot_dict["kinematics"]["tool_frames"][0]
            self._world_config = self._build_world_config()

            # Two separate solver instances, mirroring the 0.7.8 layout:
            #   - self.motion_gen        : single-target planner (with graph seeding)
            #   - self.motion_gen_batch  : ROTATE_NUM batch planner for pose screening
            # We intentionally do NOT merge these into one max_batch_size instance so
            # that single-target planning keeps its exact (unbatched) behaviour.
            self.motion_gen = MotionPlanner(self._make_cfg(max_batch_size=1))
            self.motion_gen.warmup()
            self.motion_gen_batch = BatchMotionPlanner(
                self._make_cfg(max_batch_size=CONFIGS.ROTATE_NUM)
            )
            self.motion_gen_batch.warmup()

            # Snapshot the pristine (unconstrained) tool-pose criteria of each solver
            # so a constrained plan can be reset back to it exactly afterwards.
            self._default_criteria_single = self._capture_default_criteria(self.motion_gen.trajopt_solver)
            self._default_criteria_batch = self._capture_default_criteria(self.motion_gen_batch.trajopt_solver)

        # ------------------------------------------------------------------ #
        # Construction helpers
        # ------------------------------------------------------------------ #
        @staticmethod
        def _read_knobs():
            """Read TRON2_CUROBO_* env knobs and map them to 0.8.0.

            TRON2_CUROBO_TRAJOPT_SEEDS -> num_trajopt_seeds (create arg)
            TRON2_CUROBO_MAX_ATTEMPTS  -> plan_pose(max_attempts=)
            TRON2_CUROBO_GRAPH_SEEDS   -> no direct 0.8.0 equivalent (graph seeding is
                driven by MotionPlanner's enable_graph_attempt, not a seed count). Still
                read so scripts that set it don't break, but warn that it is ignored.
            """
            graph_seeds = int(os.environ.get("TRON2_CUROBO_GRAPH_SEEDS", "1"))
            if "TRON2_CUROBO_GRAPH_SEEDS" in os.environ:
                warnings.warn(
                    "TRON2_CUROBO_GRAPH_SEEDS is deprecated and ignored under cuRobo "
                    "0.8.0 (graph seeding is controlled by enable_graph_attempt, not a "
                    "seed count).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return {
                "trajopt_seeds": int(os.environ.get("TRON2_CUROBO_TRAJOPT_SEEDS", "1")),
                "graph_seeds": graph_seeds,
                "max_attempts": int(os.environ.get("TRON2_CUROBO_MAX_ATTEMPTS", "10")),
                "interpolation_dt": float(os.environ.get("TRON2_INTERP_DT", str(1.0 / 250))),
            }

        def _build_robot_dict(self, yml_data):
            """Build the 0.8.0 robot config dict from the (converted) YAML.

            Fails loud on 0.7.x schema keys, and inlines the collision-sphere file
            because 0.8.0 does not accept a file *path* for collision_spheres.
            """
            kin = copy.deepcopy(yml_data["robot_cfg"]["kinematics"])

            if "ee_link" in kin or "retract_config" in kin.get("cspace", {}):
                raise RuntimeError(f"[Planner.py]: {_OLD_SCHEMA_HINT}\n  yml: {self.yml_path}")

            # Defensive strip of keys 0.8.0 rejects even after conversion (a None
            # asset_root_path breaks path joining; use_usd_kinematics is gone).
            kin.pop("use_usd_kinematics", None)
            if kin.get("asset_root_path") is None:
                kin.pop("asset_root_path", None)

            spheres = kin.get("collision_spheres")
            if isinstance(spheres, str):
                # 0.8.0 needs the spheres inlined; resolve a relative path against the
                # YAML's own directory (not the process cwd) so it is location-stable.
                if not os.path.isabs(spheres):
                    spheres = os.path.join(os.path.dirname(os.path.abspath(self.yml_path)), spheres)
                with open(spheres, "r") as f:
                    sphere_data = yaml.safe_load(f)
                kin["collision_spheres"] = sphere_data.get("collision_spheres", sphere_data)

            return {"kinematics": kin}

        def _build_world_config(self):
            table_top = float(os.environ.get("CUROBO_TABLE_TOP", "0.74"))
            if os.environ.get("CUROBO_NO_TABLE", "0") == "1":
                return {"cuboid": {}}
            return {
                "cuboid": {
                    "table": {
                        "dims": [0.7, 2, 0.04],  # x, y, z
                        "pose": [
                            self.robot_origion_pose.p[1],
                            0.0,
                            table_top - self.robot_origion_pose.p[2],
                            1,
                            0,
                            0,
                            0.0,
                        ],  # x, y, z, qw, qx, qy, qz
                    },
                }
            }

        def _make_cfg(self, max_batch_size):
            cfg = MotionPlannerCfg.create(
                robot=copy.deepcopy(self._robot_dict),
                scene_model=copy.deepcopy(self._world_config),
                num_trajopt_seeds=self.trajopt_seeds,
                max_batch_size=max_batch_size,
            )
            # interpolation_dt is not a create() argument in 0.8.0; override it on the
            # trajopt solver config before the solver is built so the returned
            # trajectory is sampled at 1/250 s (velocity ~= position diff / dt).
            cfg.trajopt_solver_config.interpolation_dt = self.interpolation_dt
            return cfg

        # ------------------------------------------------------------------ #
        # Constraint (partial-pose hold) helpers
        # ------------------------------------------------------------------ #
        def _capture_default_criteria(self, trajopt_solver):
            """Read the solver's pristine tool-pose criteria for self.tool_frame.

            Returns a dict of plain python values (cloned off-device) suitable for
            rebuilding a ToolPoseCriteria. Falls back to the standard default if the
            internal criteria cannot be read.
            """
            try:
                cost = trajopt_solver.core.metrics_rollout.get_cost_component_by_name("tool_pose")[0]
                st = cost._stacked_tool_pose_criteria
                idx = st.tool_frames.index(self.tool_frame)
                return {
                    "terminal_pose_axes_weight_factor": st.terminal_pose_axes_weight_factor[idx].clone().tolist(),
                    "non_terminal_pose_axes_weight_factor": st.non_terminal_pose_axes_weight_factor[idx].clone().tolist(),
                    "terminal_pose_convergence_tolerance": st.terminal_pose_convergence_tolerance[idx].clone().tolist(),
                    "non_terminal_pose_convergence_tolerance": st.non_terminal_pose_convergence_tolerance[idx].clone().tolist(),
                    "project_distance_to_goal": bool(st.project_distance_to_goal[idx, 0].clone().item()),
                }
            except Exception:
                return dict(_DEFAULT_TOOL_POSE_CRITERIA)

        def _apply_constraint(self, trajopt_solver, constraint_pose):
            """Apply a partial-pose hold to the running (non-terminal) waypoints.

            constraint_pose is the 0.7.8 hold_vec_weight (6-vec, [rot(3), trans(3)]);
            we swap the halves into cuRobo 0.8.0's [trans(3), rot(3)] axis order. The
            terminal waypoint keeps the pristine full-goal weighting so the goal is
            still reached; only the path in between is constrained.
            """
            c = [float(v) for v in constraint_pose]
            if len(c) != 6:
                raise ValueError(f"[Planner.py]: constraint_pose must be a 6-vector, got {c}")
            non_terminal = c[3:6] + c[0:3]  # [rot,trans] -> [trans,rot]
            trajopt_solver.update_tool_pose_criteria({
                self.tool_frame: ToolPoseCriteria(
                    terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    non_terminal_pose_axes_weight_factor=non_terminal,
                    project_distance_to_goal=True,
                )
            })

        def _reset_constraint(self, trajopt_solver, default_criteria):
            trajopt_solver.update_tool_pose_criteria({
                self.tool_frame: ToolPoseCriteria(**default_criteria)
            })

        # ------------------------------------------------------------------ #
        # Pose / joint helpers (unchanged transform math from 0.7.8)
        # ------------------------------------------------------------------ #
        def _target_pose_in_base(self, target_gripper_pose, arms_tag):
            """World gripper pose -> arm-base pose list [x,y,z,qw,qx,qy,qz]."""
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([
                np.array(target_gripper_pose.p),
                np.array(target_gripper_pose.q),
            ])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else:  # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            return list(target_pose_p) + list(target_pose_q)

        def _current_joint_angles(self, curr_joint_pos):
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            return [round(angle, 5) for angle in joint_angles]  # avoid the precision problem

        def _make_goal(self, pose_rows):
            """pose_rows: (N,7) cuda tensor [p(3) | q(4)] -> GoalToolPose (batch=N)."""
            return GoalToolPose.from_poses({
                self.tool_frame: CuroboPose(
                    position=pose_rows[:, :3].contiguous(),
                    quaternion=pose_rows[:, 3:].contiguous(),
                )
            })

        # ------------------------------------------------------------------ #
        # Planning
        # ------------------------------------------------------------------ #
        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):
            base_target = self._target_pose_in_base(target_gripper_pose, arms_tag)
            pose_row = torch.tensor([base_target], dtype=torch.float32).cuda()  # (1,7)
            goal = self._make_goal(pose_row)

            joint_angles = self._current_joint_angles(curr_joint_pos)
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )

            solver = self.motion_gen.trajopt_solver
            applied_constraint = False
            try:
                if constraint_pose is not None:
                    self._apply_constraint(solver, constraint_pose)
                    applied_constraint = True
                result = self.motion_gen.plan_pose(
                    goal, start_joint_states, max_attempts=self.max_attempts
                )
            finally:
                # Reset even on exception so an aborted constrained plan never leaks
                # its hold into subsequent unconstrained plans.
                if applied_constraint:
                    self._reset_constraint(solver, self._default_criteria_single)

            res_result = dict()
            if result is None or not bool(result.success.any().item()):
                res_result["status"] = "Fail"
                return res_result

            position, velocity = self._extract_single(result)
            res_result["status"] = "Success"
            res_result["position"] = position
            res_result["velocity"] = velocity
            return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Failure" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position

            Semantics: each target pose is an independent planning problem with its
            own success/failure (batch dimension), NOT a goalset ("reach any one").
            """
            num_poses = len(target_gripper_pose_list)
            batch_size = self.motion_gen_batch.batch_size
            if num_poses > batch_size:
                raise ValueError(
                    f"[Planner.py]: plan_batch got {num_poses} poses but batch planner "
                    f"is sized for {batch_size} (CONFIGS.ROTATE_NUM)."
                )

            poses_list = [self._target_pose_in_base(p, arms_tag) for p in target_gripper_pose_list]
            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()  # (num_poses,7)
            if num_poses < batch_size:
                # cuRobo captures a fixed-size CUDA graph at warmup; pad the batch to
                # batch_size (repeat the last pose) and slice results back afterwards.
                pad = poses_cuda[-1:].repeat(batch_size - num_poses, 1)
                poses_cuda = torch.cat([poses_cuda, pad], dim=0)
            goal = self._make_goal(poses_cuda)

            joint_angles = self._current_joint_angles(curr_joint_pos)
            joint_angles_cuda = torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1)
            joint_angles_cuda = joint_angles_cuda.repeat(batch_size, 1)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)

            solver = self.motion_gen_batch.trajopt_solver
            applied_constraint = False
            try:
                if constraint_pose is not None:
                    self._apply_constraint(solver, constraint_pose)
                    applied_constraint = True
                result = self.motion_gen_batch.plan_pose(
                    goal, start_joint_states, max_attempts=self.max_attempts, success_ratio=1.0
                )
            except Exception:
                return {"status": np.array(["Failure"] * num_poses, dtype=object)}
            finally:
                if applied_constraint:
                    self._reset_constraint(solver, self._default_criteria_batch)

            res_result = dict()
            if result is None:
                res_result["status"] = np.array(["Failure"] * num_poses, dtype=object)
                return res_result

            per_pose_success = result.success.any(dim=-1)[:num_poses].detach().cpu().numpy()
            status_array = np.array(
                ["Success" if s else "Failure" for s in per_pose_success], dtype=object
            )
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            position, velocity = self._extract_batch(result, num_poses)
            res_result["position"] = position
            res_result["velocity"] = velocity
            return res_result

        # ------------------------------------------------------------------ #
        # Trajectory extraction: truncate to interpolated_last_tstep + select
        # active-joint columns (0.8.0 returns the full 5000-step buffer over all
        # DOF, incl. locked gripper joints; without this the recorded trajectory
        # would carry a static tail and leak locked-joint columns).
        # ------------------------------------------------------------------ #
        def _extract_single(self, result):
            success = result.success  # (1, num_seeds)
            seed = int(torch.argmax(success[0].to(torch.int32)).item())
            last_tstep = int(result.interpolated_last_tstep.view(1, -1)[0, seed].item())
            traj = result.interpolated_trajectory.reorder(self.active_joints_name)
            position = traj.position[0, seed, :last_tstep, :].detach().cpu().numpy()
            velocity = traj.velocity[0, seed, :last_tstep, :].detach().cpu().numpy()
            return position, velocity

        def _extract_batch(self, result, num_poses):
            success = result.success  # (B, num_seeds)
            B, S = success.shape
            traj = result.interpolated_trajectory.reorder(self.active_joints_name)
            pos_full = traj.position  # (B, S, steps, dof)
            vel_full = traj.velocity
            steps, dof = pos_full.shape[2], pos_full.shape[3]

            # Pick, per problem, the seed that succeeded (argmax over the success
            # row -> first True; falls back to 0 for failed rows, which are unused).
            seeds = torch.argmax(success.to(torch.int32), dim=1)  # (B,)
            gather_idx = seeds.view(B, 1, 1, 1).expand(B, 1, steps, dof)
            pos_sel = torch.gather(pos_full, 1, gather_idx).squeeze(1)  # (B, steps, dof)
            vel_sel = torch.gather(vel_full, 1, gather_idx).squeeze(1)

            last_ts = result.interpolated_last_tstep.view(B, -1)
            last_per_pose = torch.gather(last_ts, 1, seeds.view(B, 1)).squeeze(1)  # (B,)

            real_success = success.any(dim=-1)[:num_poses]
            last_real = last_per_pose[:num_poses]
            if bool(real_success.any()):
                max_len = int(last_real[real_success].max().item())
            else:
                max_len = 1

            position = pos_sel[:num_poses, :max_len, :].detach().cpu().numpy()
            velocity = vel_sel[:num_poses, :max_len, :].detach().cpu().numpy()
            return position, velocity

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q

except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
