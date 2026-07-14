import argparse
import os

import h5py
import numpy as np


def validate_episode(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"missing episode file: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        if bool(root.attrs.get("sim", False)) is not True:
            raise AssertionError(f"{dataset_path}: expected root.attrs['sim'] is True")

        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        cam_high = root["/observations/images/cam_high"]
        cam_right_wrist = root["/observations/images/cam_right_wrist"]
        cam_left_wrist = root["/observations/images/cam_left_wrist"]

        if qpos.shape != action.shape:
            raise AssertionError(
                f"{dataset_path}: qpos shape {qpos.shape} != action shape {action.shape}"
            )
        if qpos.ndim != 2:
            raise AssertionError(
                f"{dataset_path}: expected qpos.ndim == 2, got {qpos.ndim}"
            )
        if qpos.shape[1] != 16:
            raise AssertionError(
                f"{dataset_path}: expected qpos.shape[1] == 16, got {qpos.shape[1]}"
            )
        if action.shape[1] != 16:
            raise AssertionError(
                f"{dataset_path}: expected action.shape[1] == 16, got {action.shape[1]}"
            )
        if len(cam_high) != len(qpos):
            raise AssertionError(
                f"{dataset_path}: cam_high length {len(cam_high)} != qpos length {len(qpos)}"
            )
        if len(cam_right_wrist) != len(qpos):
            raise AssertionError(
                f"{dataset_path}: cam_right_wrist length {len(cam_right_wrist)} "
                f"!= qpos length {len(qpos)}"
            )
        if len(cam_left_wrist) != len(qpos):
            raise AssertionError(
                f"{dataset_path}: cam_left_wrist length {len(cam_left_wrist)} "
                f"!= qpos length {len(qpos)}"
            )
        if not np.allclose(qpos[1:], action[:-1], atol=1e-6, rtol=0.0):
            raise AssertionError(
                f"{dataset_path}: expected qpos[1:] to equal action[:-1]"
            )

        return len(qpos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_episodes", type=int, required=True)
    args = parser.parse_args()

    episode_lengths = []
    for episode_idx in range(args.num_episodes):
        dataset_path = os.path.join(
            args.dataset_dir,
            f"episode_{episode_idx}.hdf5",
        )
        episode_lengths.append(validate_episode(dataset_path))

    print(f"validated_episodes: {len(episode_lengths)}")
    print(f"min_episode_len: {min(episode_lengths)}")
    print(f"max_episode_len: {max(episode_lengths)}")
    print(f"mean_episode_len: {float(np.mean(episode_lengths))}")


if __name__ == "__main__":
    main()