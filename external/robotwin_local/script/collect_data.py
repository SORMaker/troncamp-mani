import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def _handle_replay_result(TASK_ENV, args, episode_idx, seed_list):
    # Stock behavior: assert the replayed episode still succeeds. Replay can diverge
    # from the search-phase outcome due to SAPIEN physics non-determinism (severe for
    # hard tasks like hanging_mug), which crashes the whole run. With
    # ROBOTWIN_SKIP_FAILED_REPLAY=1 we instead delete the just-saved bad episode and
    # continue, so good seeds are not lost to one bad replay. The caller renumbers via
    # a post-pass (merge), so a hole here is tolerated.
    if TASK_ENV.check_success():
        return
    if os.environ.get('ROBOTWIN_SKIP_FAILED_REPLAY', '0') != '1':
        raise AssertionError('Collect Error')
    data_path = os.path.join(args['save_path'], 'data', 'episode' + str(episode_idx) + '.hdf5')
    if os.path.exists(data_path):
        os.remove(data_path)
    print('[skip-replay] episode ' + str(episode_idx) + ' failed replay; dropped, continuing '
          '(leaves a numbering hole — renumber before process_data)')


def run(TASK_ENV, args):
    # Seed search starts at epid=0 by default. ROBOTWIN_SEED_START lets parallel
    # collection workers explore disjoint seed ranges (each writes its own save dir),
    # so N workers find ~N× distinct successful seeds concurrently instead of all
    # re-discovering the same low seeds. Only affects the search start, not behavior.
    epid, suc_num, fail_num, seed_list = int(os.environ.get("ROBOTWIN_SEED_START", "0")), 0, 0, []
    # Single-pass collection flag (function scope so the replay-phase gate below can
    # see it). See the search loop for the rationale. Only meaningful when we actually
    # run the search pass: use_seed=True skips search and goes straight to replay, so a
    # single-pass flag there would disable replay too and collect nothing — force it off.
    _save_during_search = (
        os.environ.get("ROBOTWIN_SAVE_DURING_SEARCH", "0") == "1" and not args["use_seed"]
    )

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== curobo RNG warmup (do this BEFORE any real episode) ===========
    # Building the curobo planner consumes the numpy global RNG mid-call, so the very
    # first setup_demo in a process produces a frozen, seed-independent scene; every later
    # setup_demo on the same env then sees the true per-seed RNG. In two-phase collection
    # the search pass would build the planner on episode 0 (polluting that scene) while the
    # replay pass re-runs the same seed on an already-warm planner (clean scene). The two
    # passes then disagree on which arm the expert uses, and replay indexes an empty joint
    # path (left_joint_path[left_cnt] on []), crashing with an uncaught IndexError. Warm the
    # planner once here on a throwaway scene so search and replay agree. play_once is never
    # called, so nothing is written. See tools/web_demo_render.py warmup_curobo() for the
    # same fix. This must run before the use_seed branch too (use_seed goes straight to
    # replay), which is why it lives at the top of run().
    try:
        TASK_ENV.setup_demo(now_ep_num=0, seed=0, **args)
    except Exception as _warm_e:  # noqa: BLE001 — planner is cached before load_actors, so a late failure is harmless
        print(f"[warmup] curobo warmup setup_demo raised (harmless, planner is cached): {_warm_e}")
    finally:
        try:
            TASK_ENV.close_env()
        except Exception:
            pass

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True
        # Single-pass collection: capture observations DURING the successful planning
        # run instead of replaying it afterward. Required for tasks whose replay is
        # non-deterministic (hanging_mug: the saved joint path replays but the delicate
        # hang fails check_success ~100% of the time, so the stock two-phase flow yields
        # ~0 episodes). With ROBOTWIN_SAVE_DURING_SEARCH=1 we save_data in the search
        # pass and write the HDF5 immediately on success, skipping the replay phase.
        if _save_during_search:
            args["save_data"] = True
            print("\033[93m" + "[ROBOTWIN_SAVE_DURING_SEARCH=1: single-pass save]" + "\033[0m")
        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                _sp_info = TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    if _save_during_search:
                        # Observations were captured during this successful planning run;
                        # write the HDF5 now (the stock replay phase would re-run and, for
                        # hanging_mug, fail check_success ~100%). Renumber-safe: episodes are
                        # saved under suc_num which increments only on success.
                        TASK_ENV.merge_pkl_to_hdf5_video()
                        TASK_ENV.remove_data_cache()
                        # Single-pass also skips the replay phase that normally writes
                        # scene_info.json (lines ~251-264). gen_episode_instructions.sh
                        # (run at the end of main) needs it, so write this episode's
                        # play_once info here, keyed by suc_num to match episode{suc_num}.hdf5.
                        _sp_path = os.path.join(args["save_path"], "scene_info.json")
                        _sp_db = {}
                        if os.path.exists(_sp_path):
                            with open(_sp_path, "r", encoding="utf-8") as _spf:
                                _sp_db = json.load(_spf)
                        _sp_db[f"episode_{suc_num}"] = _sp_info
                        with open(_sp_path, "w", encoding="utf-8") as _spf:
                            json.dump(_sp_db, _spf, ensure_ascii=False, indent=4)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"] and not _save_during_search:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            try:
                info = TASK_ENV.play_once()
            except Exception as e:
                # A replay crash here is almost always the search/replay scene divergence
                # described in the warmup note above (an uncaught IndexError from indexing an
                # empty joint path when replay picks a different arm than search). Never let it
                # take down the whole run silently.
                print(f"\033[91m[replay-crash] episode {episode_idx}: {e}\033[0m")
                # Best-effort cleanup so a crashed episode leaks no sapien scene or partial
                # cache (the normal close_env/remove_data_cache at the loop tail is skipped
                # when we bail out here). Target this episode's cache dir directly rather than
                # self.folder_path, which may still point at the previous episode.
                try:
                    TASK_ENV.close_env(clear_cache=True)
                except Exception:
                    pass
                _crash_cache = os.path.join(args["save_path"], ".cache", f"episode{episode_idx}")
                if os.path.isdir(_crash_cache):
                    import shutil as _shutil
                    _shutil.rmtree(_crash_cache, ignore_errors=True)
                if os.environ.get("ROBOTWIN_SKIP_FAILED_REPLAY", "0") == "1":
                    bad_path = os.path.join(args["save_path"], "data", f"episode{episode_idx}.hdf5")
                    if os.path.exists(bad_path):
                        os.remove(bad_path)
                    # NOTE: dropping an episode here leaves a numbering hole (episodeN missing).
                    # ACT process_data expects contiguous episode0..episode{num-1}, so a merge/
                    # renumber post-pass is required before conversion (same tolerated behavior as
                    # the stock _handle_replay_result skip). The default single-pass collection
                    # (ROBOTWIN_SAVE_DURING_SEARCH=1) never holes and needs no post-pass.
                    print(f"[skip-replay] episode {episode_idx} dropped after crash; continuing "
                          "(leaves a numbering hole — renumber before process_data)")
                    continue
                raise RuntimeError(
                    f"Replay of episode {episode_idx} crashed ({e}). This is usually the curobo "
                    "RNG search/replay scene divergence. Re-run with ROBOTWIN_SAVE_DURING_SEARCH=1 "
                    "(single-pass save, immune to this and the default) or ROBOTWIN_SKIP_FAILED_REPLAY=1 "
                    "(drop the bad episode and continue; leaves a hole to renumber before process_data)."
                ) from e
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            _handle_replay_result(TASK_ENV, args, episode_idx, seed_list)

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
