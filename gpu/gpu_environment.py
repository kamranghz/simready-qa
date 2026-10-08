"""MJWarp GPU environment for the SimReadyQA 12-DOF humanoid."""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

JOINT_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
]

CANONICAL_QPOS = np.array(
    [0.20, 0.15, 0.30, 0.20, -0.15, 0.30, 0.10, 0.20, -0.05, 0.10, 0.20, -0.05],
    dtype=np.float32,
)
CANONICAL_CTRL = np.array(
    [0.08, 0.04, 0.05, 0.08, -0.04, 0.05, 0.04, 0.03, 0.02, 0.04, 0.03, 0.02],
    dtype=np.float32,
)
NUM_JOINTS = 12
NUM_STEPS = 200


def require_cuda() -> wp.context.Device:
    wp.init()
    if not wp.is_cuda_available():
        device = wp.get_device()
        raise RuntimeError(
            f"GPU backend requires a CUDA device; Warp selected {device}."
        )
    device = wp.get_device("cuda:0")
    wp.set_device(device)
    return device


def print_gpu_environment() -> wp.context.Device:
    device = require_cuda()
    print("GPU environment")
    print(f"  warp-lang = {getattr(wp, '__version__', wp.config.version)}")
    mjw_version = getattr(mjw, "__version__", "unknown")
    print(f"  mujoco-warp = {mjw_version}")
    print(f"  mujoco python = {mujoco.__version__}")
    print(f"  numpy = {np.__version__}")
    print(f"  Warp device = {device}")
    is_cuda = bool(getattr(device, "is_cuda", True))
    print(f"  device is CUDA = {is_cuda}")
    print(f"  NVIDIA GPU name = {getattr(device, 'name', str(device))}")
    return device


def load_mjcf(xml_path: str):
    mjm = mujoco.MjModel.from_xml_path(xml_path)
    mjd = mujoco.MjData(mjm)
    if mjm.nq != NUM_JOINTS or mjm.nu != NUM_JOINTS:
        raise RuntimeError(f"Expected nq=nu={NUM_JOINTS}, got nq={mjm.nq} nu={mjm.nu}")
    return mjm, mjd


def make_warp_sim(mjm, mjd, nworld: int, nconmax: int = 24, njmax: int = 80):
    model = mjw.put_model(mjm)
    data = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
    return model, data


def set_world_state(data, qpos_batch: np.ndarray, qvel_batch: np.ndarray, ctrl_batch: np.ndarray):
    data.qpos = wp.from_numpy(np.ascontiguousarray(qpos_batch, dtype=np.float32), dtype=wp.float32)
    data.qvel = wp.from_numpy(np.ascontiguousarray(qvel_batch, dtype=np.float32), dtype=wp.float32)
    data.ctrl = wp.from_numpy(np.ascontiguousarray(ctrl_batch, dtype=np.float32), dtype=wp.float32)


def get_qpos_qvel(data) -> tuple[np.ndarray, np.ndarray]:
    return data.qpos.numpy(), data.qvel.numpy()


def default_xml() -> str:
    return str(Path(__file__).resolve().parents[1] / "assets" / "humanoid_12dof.xml")
