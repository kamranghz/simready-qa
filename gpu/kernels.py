"""Custom NVIDIA Warp kernel used by SimReadyQA GPU validation."""

import warp as wp

NUM_JOINTS = 12


@wp.kernel
def validate_rollouts(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    max_abs_qpos: float,
    max_abs_qvel: float,
    success: wp.array(dtype=wp.int32),
):
    """One thread per world: finite-state and safety-limit checks."""
    env_id = wp.tid()
    ok = int(1)
    for i in range(NUM_JOINTS):
        q = qpos[env_id, i]
        v = qvel[env_id, i]
        if wp.isnan(q) or wp.isinf(q) or wp.isnan(v) or wp.isinf(v):
            ok = int(0)
        if wp.abs(q) > max_abs_qpos or wp.abs(v) > max_abs_qvel:
            ok = int(0)
    success[env_id] = ok
