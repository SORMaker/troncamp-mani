import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    # 存成 (n, max_len) uint8 数组:不能用 VLEN bytes(padded 的 \0 触发 h5py "embedded NULL")。
    return np.array([np.frombuffer(d, np.uint8) for d in padded_data], dtype=np.uint8), max_len


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        num_frames = left_gripper_all.shape[0]
        if num_frames < 2:
            raise ValueError(
                f"episode {i} has only {num_frames} frame(s); at least 2 frames are required"
            )

        states = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []

        for j in range(num_frames):
            state = np.concatenate(
                (
                    left_arm_all[j],
                    [left_gripper_all[j]],
                    right_arm_all[j],
                    [right_gripper_all[j]],
                ),
                axis=0,
            ).astype(np.float32)

            if state.shape != (16,):
                raise ValueError(
                    f"episode {i}, frame {j}: expected state shape (16,), got {state.shape}"
                )

            states.append(state)

            if j < num_frames - 1:
                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(
                    np.frombuffer(camera_high_bits, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if camera_high is None:
                    raise ValueError(f"episode {i}, frame {j}: failed to decode head_camera")
                cam_high.append(cv2.resize(camera_high, (640, 480)))

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(
                    np.frombuffer(camera_right_wrist_bits, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if camera_right_wrist is None:
                    raise ValueError(f"episode {i}, frame {j}: failed to decode right_camera")
                cam_right_wrist.append(cv2.resize(camera_right_wrist, (640, 480)))

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(
                    np.frombuffer(camera_left_wrist_bits, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if camera_left_wrist is None:
                    raise ValueError(f"episode {i}, frame {j}: failed to decode left_camera")
                cam_left_wrist.append(cv2.resize(camera_left_wrist, (640, 480)))

        states = np.asarray(states, dtype=np.float32)
        qpos = states[:-1]
        actions = states[1:]

        left_arm_dim = np.full(
            qpos.shape[0],
            left_arm_all.shape[1],
            dtype=np.int32,
        )
        right_arm_dim = np.full(
            qpos.shape[0],
            right_arm_all.shape[1],
            dtype=np.int32,
        )

        if qpos.shape != actions.shape:
            raise AssertionError(
                f"episode {i}: qpos shape {qpos.shape} != action shape {actions.shape}"
            )

        if qpos.shape[0] != len(cam_high):
            raise AssertionError(
                f"episode {i}: qpos length {qpos.shape[0]} != cam_high length {len(cam_high)}"
            )

        if qpos.shape[0] != len(cam_right_wrist):
            raise AssertionError(
                f"episode {i}: qpos length {qpos.shape[0]} != "
                f"cam_right_wrist length {len(cam_right_wrist)}"
            )

        if qpos.shape[0] != len(cam_left_wrist):
            raise AssertionError(
                f"episode {i}: qpos length {qpos.shape[0]} != "
                f"cam_left_wrist length {len(cam_left_wrist)}"
            )

        if not np.allclose(qpos[1:], actions[:-1], atol=1e-6, rtol=0.0):
            raise AssertionError(
                f"episode {i}: expected qpos[1:] to equal action[:-1]"
            )
        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.attrs["sim"] = True
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            # JPEG 压缩存储(避免 raw uint8 炸盘 ~40×;loader 端 imdecode 还原,eval 读 live sim 不受影响)
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc)
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc)
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
