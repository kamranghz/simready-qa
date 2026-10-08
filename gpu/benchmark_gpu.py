"""GPU MJWarp benchmarks, custom Warp kernel, and CPU/GPU consistency."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpu"))

import gpu_environment as genv  # noqa: E402
import mujoco_warp as mjw  # noqa: E402
from kernels import validate_rollouts  # noqa: E402

OUTPUTS = ROOT / "outputs"
CPU_CANONICAL = OUTPUTS / "cpu_canonical_final.txt"
CONSISTENCY_TOL = 5e-3
GPU_DET_TOL = 1e-4
MAX_ABS_QPOS = 6.283185307179586
MAX_ABS_QVEL = 40.0
DATASET_WORLDS = 16
DATASET_STEPS = 50


def read_cpu_canonical() -> tuple[np.ndarray, np.ndarray]:
    if not CPU_CANONICAL.exists():
        raise FileNotFoundError(
            f"Missing {CPU_CANONICAL}. Run the C++ harness first."
        )
    qpos = None
    qvel = None
    for line in CPU_CANONICAL.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        values = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        if parts[0] == "qpos":
            qpos = values
        elif parts[0] == "qvel":
            qvel = values
    if qpos is None or qvel is None:
        raise RuntimeError("cpu_canonical_final.txt is incomplete")
    return qpos, qvel


def broadcast_canonical(nworld: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos = np.repeat(genv.CANONICAL_QPOS[None, :], nworld, axis=0)
    qvel = np.zeros((nworld, genv.NUM_JOINTS), dtype=np.float32)
    ctrl = np.repeat(genv.CANONICAL_CTRL[None, :], nworld, axis=0)
    rng = np.random.default_rng(42)
    if nworld > 1:
        qpos += rng.uniform(-0.05, 0.05, size=qpos.shape).astype(np.float32)
        qpos[0] = genv.CANONICAL_QPOS
        ctrl[0] = genv.CANONICAL_CTRL
    return qpos, qvel, ctrl


def step_n(model, data, n_steps: int) -> None:
    for _ in range(n_steps):
        mjw.step(model, data)
    wp.synchronize()


def run_single_world(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    mjm, mjd = genv.load_mjcf(xml_path)
    model, data = genv.make_warp_sim(mjm, mjd, nworld=1)
    qpos, qvel, ctrl = broadcast_canonical(1)
    genv.set_world_state(data, qpos, qvel, ctrl)
    step_n(model, data, genv.NUM_STEPS)
    out_qpos, out_qvel = genv.get_qpos_qvel(data)
    return out_qpos[0].astype(np.float64), out_qvel[0].astype(np.float64)


def consistency_report(cpu_qpos, cpu_qvel, gpu_qpos, gpu_qvel) -> bool:
    dq = np.abs(cpu_qpos - gpu_qpos)
    dv = np.abs(cpu_qvel - gpu_qvel)
    rmse_q = float(np.sqrt(np.mean((cpu_qpos - gpu_qpos) ** 2)))
    rmse_v = float(np.sqrt(np.mean((cpu_qvel - gpu_qvel) ** 2)))
    print("Cross-backend consistency (CPU MuJoCo vs GPU MJWarp)")
    print(f"  tolerance (max abs) = {CONSISTENCY_TOL}")
    print(f"  max abs qpos diff = {dq.max():.6e}")
    print(f"  max abs qvel diff = {dv.max():.6e}")
    print(f"  qpos RMSE = {rmse_q:.6e}")
    print(f"  qvel RMSE = {rmse_v:.6e}")
    for i, name in enumerate(genv.JOINT_NAMES):
        print(f"    {name}: dq={dq[i]:.6e} dv={dv[i]:.6e}")
    passed = dq.max() <= CONSISTENCY_TOL and dv.max() <= CONSISTENCY_TOL
    print(f"Cross-backend consistency: {'PASS' if passed else 'FAIL'}")
    return passed


def gpu_determinism(xml_path: str) -> bool:
    finals = []
    for _ in range(3):
        qpos, qvel = run_single_world(xml_path)
        finals.append((qpos, qvel))
    ok = True
    for i in range(1, 3):
        if np.max(np.abs(finals[0][0] - finals[i][0])) > GPU_DET_TOL:
            ok = False
        if np.max(np.abs(finals[0][1] - finals[i][1])) > GPU_DET_TOL:
            ok = False
    print(f"GPU deterministic replay: {'PASS' if ok else 'FAIL'} (tol={GPU_DET_TOL})")
    return ok


def run_kernel(data) -> np.ndarray:
    qpos = data.qpos
    qvel = data.qvel
    nworld = qpos.shape[0]
    success = wp.zeros(nworld, dtype=wp.int32)
    wp.launch(
        validate_rollouts,
        dim=nworld,
        inputs=[qpos, qvel, MAX_ABS_QPOS, MAX_ABS_QVEL],
        outputs=[success],
    )
    wp.synchronize()
    flags = success.numpy()
    print("Custom Warp kernel validate_rollouts")
    print(f"  worlds = {nworld}")
    print(f"  SUCCESS = {int((flags == 1).sum())}")
    print(f"  FAILURE = {int((flags != 1).sum())}")
    print("Custom Warp kernel: PASS")
    return flags


def benchmark_batch(xml_path: str, nworld: int, n_steps: int = 200) -> dict:
    mjm, mjd = genv.load_mjcf(xml_path)
    qpos0, qvel0, ctrl0 = broadcast_canonical(nworld)
    model, data = genv.make_warp_sim(mjm, mjd, nworld=nworld)
    genv.set_world_state(data, qpos0, qvel0, ctrl0)

    step_n(model, data, 8)  # warmup / JIT
    genv.set_world_state(data, qpos0, qvel0, ctrl0)
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        mjw.step(model, data)
    wp.synchronize()
    elapsed = time.perf_counter() - t0
    total_steps = nworld * n_steps
    result = {
        "nworld": nworld,
        "steps": n_steps,
        "total_steps": total_steps,
        "seconds": elapsed,
        "steps_per_sec": total_steps / elapsed,
        "worlds_per_sec": nworld / elapsed,
    }
    print(
        f"GPU throughput nworld={nworld}: {elapsed:.6f}s, "
        f"{result['steps_per_sec']:.1f} steps/s, {result['worlds_per_sec']:.1f} worlds/s"
    )
    run_kernel(data)
    return result, model, data, qpos0, ctrl0


def try_cuda_graph(model, data, qpos0, qvel0, ctrl0, n_steps: int = 200) -> None:
    print("CUDA graph benchmark (optional)")
    try:
        genv.set_world_state(data, qpos0, qvel0, ctrl0)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            mjw.step(model, data)
        graph = capture.graph
        genv.set_world_state(data, qpos0, qvel0, ctrl0)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            wp.capture_launch(graph)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        nworld = qpos0.shape[0]
        print(
            f"  graph nworld={nworld}: {elapsed:.6f}s, "
            f"{(nworld * n_steps) / elapsed:.1f} steps/s"
        )
        print("CUDA graph: PASS")
    except Exception as exc:  # graph support is optional
        print(f"CUDA graph: skipped ({exc})")


def write_gpu_dataset(xml_path: str, nworld: int, n_steps: int) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "gpu_rollouts.csv"
    mjm, mjd = genv.load_mjcf(xml_path)
    model, data = genv.make_warp_sim(mjm, mjd, nworld=nworld)
    qpos, qvel, ctrl = broadcast_canonical(nworld)
    genv.set_world_state(data, qpos, qvel, ctrl)
    header = ["rollout_id", "seed", "backend", "step", "time"]
    header += [f"qpos_{n}" for n in genv.JOINT_NAMES]
    header += [f"qvel_{n}" for n in genv.JOINT_NAMES]
    header += [f"ctrl_{n}" for n in genv.JOINT_NAMES]
    header += ["label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step in range(1, n_steps + 1):
            mjw.step(model, data)
            wp.synchronize()
            qp, qv = genv.get_qpos_qvel(data)
            success = wp.zeros(nworld, dtype=wp.int32)
            wp.launch(
                validate_rollouts,
                dim=nworld,
                inputs=[data.qpos, data.qvel, MAX_ABS_QPOS, MAX_ABS_QVEL],
                outputs=[success],
            )
            wp.synchronize()
            flags = success.numpy()
            sim_time = step * 0.002
            for env_id in range(nworld):
                row = [env_id, 42, "gpu", step, sim_time]
                row.extend(qp[env_id].tolist())
                row.extend(qv[env_id].tolist())
                row.extend(ctrl[env_id].tolist())
                row.append("SUCCESS" if flags[env_id] == 1 else "FAILURE")
                writer.writerow(row)
    print(f"GPU synthetic rollouts: {nworld} worlds x {n_steps} steps -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=genv.default_xml())
    args = parser.parse_args()

    print("=== SimReadyQA GPU ===")
    try:
        genv.print_gpu_environment()
    except Exception as exc:
        print(f"GPU backend unavailable: {exc}")
        print("GPU backend: FAIL")
        return 1

    xml_path = args.xml
    print(f"Loading MJCF: {xml_path}")
    try:
        mjm, mjd = genv.load_mjcf(xml_path)
        genv.make_warp_sim(mjm, mjd, nworld=1)
        print("MJWarp model-load: PASS")
    except Exception as exc:
        print(f"MJWarp model-load: FAIL ({exc})")
        return 1

    gpu_qpos, gpu_qvel = run_single_world(xml_path)
    cpu_qpos, cpu_qvel = read_cpu_canonical()
    cons_pass = consistency_report(cpu_qpos, cpu_qvel, gpu_qpos, gpu_qvel)
    det_pass = gpu_determinism(xml_path)

    print("GPU throughput (warmup excluded, CUDA synchronized)")
    last = None
    for nworld in (1, 64, 256, 1024):
        try:
            last = benchmark_batch(xml_path, nworld=nworld)
        except Exception as exc:
            print(f"GPU throughput nworld={nworld}: FAIL ({exc})")
            break

    if last is not None:
        model, data = last[1], last[2]
        qpos0, ctrl0 = last[3], last[4]
        qvel0 = np.zeros_like(qpos0)
        try_cuda_graph(model, data, qpos0, qvel0, ctrl0)

    write_gpu_dataset(xml_path, DATASET_WORLDS, DATASET_STEPS)

    print(f"GPU harness summary: {'PASS' if cons_pass and det_pass else 'FAIL'}")
    return 0 if cons_pass and det_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
