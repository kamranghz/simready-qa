# SimReadyQA
### C++ / CUDA Simulation QA, Deterministic Rollouts & GPU Batched MuJoCo Harness

SimReadyQA validates a fixed-base 12-DOF full-body humanoid, then runs the same plant on two backends: **C++ MuJoCo on CPU** (reference) and **Python MJWarp on NVIDIA CUDA** (batched GPU). It measures asset QA, determinism, synthetic rollouts, regression, throughput, a custom Warp kernel, and CPU/GPU numerical consistency.

This is not a walking robot, not an RL trainer, and not a hardware-in-the-loop stack.

## Motivation

Articulated assets should not reach evaluation with missing joints, invalid inertias, or irreproducible physics. GPU batching should not be trusted until it is checked against a CPU reference on the same MJCF.

## System Architecture

```
MJCF / URDF 12-DOF humanoid
        ↓
Asset QA Gate
        ├── C++ MuJoCo CPU  (reset / step / state)
        └── MJWarp + NVIDIA Warp CUDA
                ↓
        Deterministic replay
        Synthetic rollouts
        Regression + throughput
        Custom Warp validation kernel
        CPU ↔ GPU consistency
```

## Backends

| | CPU reference | GPU |
|---|---|---|
| Language | C++17 | Python |
| Physics | MuJoCo 3.13.0 (`mj_step`) | `mujoco-warp` 3.13.0 |
| Device | host CPU | NVIDIA Warp CUDA |
| Tested | MSVC Release | RTX 4090, `cuda:0`, warp-lang 1.17.0 |

GPU **PASS requires a CUDA device**. The scripts do not silently treat CPU Warp as GPU.

## 12-DOF Humanoid

Pelvis and torso are **welded to the world** (no floating base). Neck and head are welded. Twelve hinge motors:

```
world
└── pelvis
    ├── torso → neck → head
    │     ├── left_shoulder → left_upper_arm → left_forearm
    │     └── right_shoulder → right_upper_arm → right_forearm
    ├── left_thigh → left_shank → left_foot
    └── right_thigh → right_shank → right_foot
```

| Joint | Axis |
|---|---|
| `left_shoulder_pitch` / `right_shoulder_pitch` | +Y |
| `left_shoulder_roll` / `right_shoulder_roll` | +X |
| `left_elbow` / `right_elbow` | +Y |
| `left_hip_pitch` / `right_hip_pitch` | +Y |
| `left_knee` / `right_knee` | +Y |
| `left_ankle_pitch` / `right_ankle_pitch` | +Y |

Motors use the same names with `_motor` and `gear=1` (`ctrl` in N·m).

Visual geoms (`contype=0`, `density=0`): capsules, joint housings, feet, head.  
Collision geoms are **capsules/spheres only** (MJWarp-stable primitives). The floor is a collidable plane with friction; each foot uses heel and ball spheres so stance cannot roll about the ankle.

Cylinder inertia about COM:

```
I_longitudinal = (1/2) * m * r^2
I_transverse   = (1/12) * m * L^2 + (1/4) * m * r^2
```

| Body | Mass (kg) | COM (m) | diaginertia (kg·m²) |
|---|---|---|---|
| pelvis | 8.0 | (0,0,0) | 0.04187, 0.03120, 0.05387 |
| torso | 12.0 | (0,0,0) | 0.1808, 0.1348, 0.1108 |
| neck | 0.8 | (0,0,0) | 0.00045, 0.00045, 0.0002 |
| head | 4.0 | (0,0,0) | 0.01296, 0.01296, 0.01296 |
| L/R shoulder | 0.4 | (0,0,0) | 0.000373, 0.000320, 0.000373 |
| L/R upper arm | 1.5 | (0,0,−0.14) | 0.0104, 0.0104, 0.0012 |
| L/R forearm | 0.9 | (0,0,−0.125) | 0.00489, 0.00489, 0.000405 |
| L/R thigh | 6.0 | (0,0,−0.21) | 0.09274, 0.09274, 0.009075 |
| L/R shank | 3.0 | (0,0,−0.20) | 0.0412, 0.0412, 0.0024 |
| L/R foot | 0.8 | (0.07,0,−0.015) | 0.000593, 0.002833, 0.003093 |

## MJCF and URDF

`assets/humanoid_12dof.xml` is what both backends load.  
`assets/humanoid_12dof.urdf` is an equivalent tree; the runtime does not compile URDF.

## Asset QA

CPU checks 12 joints/actuators, required bodies, timestep 0.002, masses, COM, inertia. Failure skips rollouts. GPU load is checked separately in `gpu/benchmark_gpu.py`.

## Environment API

C++ `HumanoidEnv`: `reset(state, action)`, `step(action)`, `getState()` over 12 DOF.  
GPU: `mjw.put_model` / `put_data(nworld=...)` / `mjw.step`, then `d.qpos.numpy()`.

## Deterministic Replay

CPU: three identical canonical resets, match at `1e-12`.  
GPU: three identical single-world runs, match at `1e-4` (float32 device).

## Custom Warp Kernel

`gpu/kernels.py` `validate_rollouts` runs one CUDA thread per world, checks finite `qpos`/`qvel` and safety limits, writes SUCCESS/FAILURE. It is launched after each GPU batch.

## Synthetic Rollouts

CPU: 100 rollouts × 200 steps, seed 42, `outputs/rollouts.csv` (`backend=cpu`).  
GPU: 16 worlds × 50 steps, `outputs/gpu_rollouts.csv` (`backend=gpu`).  
SUCCESS = finite state, `|qpos|≤2π`, `|qvel|≤40`, completed duration. Not task success.

## Regression

`assets/regression_baseline.txt` stores the canonical 12-DOF final state. CPU max abs error vs baseline must be `≤1e-9`.

## Throughput (measured)

CPU Release, in-memory, 100 sequential rollouts × 200 steps:

- wall-clock **0.103157 s**
- **193879.4 steps/s**, **969.4 rollouts/s**

GPU MJWarp on RTX 4090, 200 steps, warmup excluded, `wp.synchronize()`:

| nworld | wall-clock | steps/s | worlds/s |
|---|---|---|---|
| 1 | 1.734 s | 115.3 | 0.6 |
| 64 | 1.722 s | 7434.4 | 37.2 |
| 256 | 1.746 s | 29315.8 | 146.6 |
| 1024 | 1.757 s | 116589.3 | 582.9 |

CUDA graph, nworld=1024: **0.190 s**, **1,080,034 steps/s**.

Single-world GPU is slower than CPU (launch overhead). Graph-captured 1024-world stepping is about **5.6×** the CPU sequential steps/s figure above. That comparison is aggregate throughput, not a claim that one GPU world is faster than one CPU world.

## CPU ↔ GPU Consistency

Same MJCF, same canonical `qpos`/`qvel`/`ctrl`, 200 steps. Tolerance: max abs **5e-3** (not bitwise). Measured: max `|Δqpos|=1.53e-6`, max `|Δqvel|=8.38e-6`, qpos RMSE `5.8e-7`. This is **cross-backend consistency**, not sim-to-real.

## Reference Trace

`assets/example_reference_trace.csv` is a **synthetic** CPU trace. Hardware traces are not included. The comparator would accept a future CSV with the same columns.

## Build (CPU)

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DMUJOCO_DIR=C:\path\to\mujoco-3.13.0
cmake --build build --config Release
.\build\Release\simready_qa.exe
```

## GPU setup

Tested: Python 3.11, `mujoco==3.13.0`, `mujoco-warp==3.13.0`, `warp-lang==1.17.0`, `numpy==2.4.6`, CUDA driver 13.3 / Warp CUDA toolkit 12.9.

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe gpu\benchmark_gpu.py
```

Run the C++ harness first so `outputs/cpu_canonical_final.txt` exists.

## Visualize

```bat
<MUJOCO_DIR>\bin\simulate.exe assets\humanoid_12dof.xml
```

## Repository Structure

```
SimReadyQA/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── requirements-gpu.txt
├── assets/
│   ├── humanoid_12dof.xml
│   ├── humanoid_12dof.urdf
│   ├── regression_baseline.txt
│   └── example_reference_trace.csv
├── include/simready/environment.h
├── src/environment.cpp
├── src/main.cpp
└── gpu/
    ├── gpu_environment.py
    ├── kernels.py
    └── benchmark_gpu.py
```

`build/`, `outputs/`, and `.venv/` are generated.
