#include "simready/environment.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace simready {

const char* kJointNames[kNumJoints] = {
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "left_hip_pitch", "left_knee", "left_ankle_pitch",
    "right_hip_pitch", "right_knee", "right_ankle_pitch"};

const char* kActuatorNames[kNumJoints] = {
    "left_shoulder_pitch_motor", "left_shoulder_roll_motor", "left_elbow_motor",
    "right_shoulder_pitch_motor", "right_shoulder_roll_motor", "right_elbow_motor",
    "left_hip_pitch_motor", "left_knee_motor", "left_ankle_pitch_motor",
    "right_hip_pitch_motor", "right_knee_motor", "right_ankle_pitch_motor"};

const char* kRequiredBodies[] = {
    "pelvis", "torso", "head",
    "left_shoulder", "left_upper_arm", "left_forearm",
    "right_shoulder", "right_upper_arm", "right_forearm",
    "left_thigh", "left_shank", "left_foot",
    "right_thigh", "right_shank", "right_foot"};
const int kNumRequiredBodies = 15;

const char* kMovingBodies[] = {
    "left_shoulder", "left_upper_arm", "left_forearm",
    "right_shoulder", "right_upper_arm", "right_forearm",
    "left_thigh", "left_shank", "left_foot",
    "right_thigh", "right_shank", "right_foot"};
const int kNumMovingBodies = 12;

namespace {

void printCheck(const char* name, bool passed) {
  std::printf("  %s: %s\n", name, passed ? "PASS" : "FAIL");
}

bool isFinitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool inertiaLooksPhysical(double ixx, double iyy, double izz) {
  if (!isFinitePositive(ixx) || !isFinitePositive(iyy) || !isFinitePositive(izz)) {
    return false;
  }
  const double slack = 1e-9;
  return (ixx + iyy + slack >= izz) && (iyy + izz + slack >= ixx) && (izz + ixx + slack >= iyy);
}

}  // namespace

bool HumanoidEnv::load(const char* model_path, char* error, int error_size) {
  close();
  model = mj_loadXML(model_path, nullptr, error, error_size);
  if (!model) {
    return false;
  }
  data = mj_makeData(model);
  if (!data) {
    std::snprintf(error, static_cast<size_t>(error_size), "failed to allocate mjData");
    mj_deleteModel(model);
    model = nullptr;
    return false;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    actuator_ids[i] = mj_name2id(model, mjOBJ_ACTUATOR, kActuatorNames[i]);
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, kJointNames[i]);
    qpos_index[i] = (joint_id >= 0) ? model->jnt_qposadr[joint_id] : -1;
    qvel_index[i] = (joint_id >= 0) ? model->jnt_dofadr[joint_id] : -1;
  }
  return true;
}

void HumanoidEnv::close() {
  if (data) {
    mj_deleteData(data);
    data = nullptr;
  }
  if (model) {
    mj_deleteModel(model);
    model = nullptr;
  }
}

void HumanoidEnv::reset(const JointState& state, const Action& action) {
  mj_resetData(model, data);
  for (int i = 0; i < kNumJoints; ++i) {
    data->qpos[qpos_index[i]] = state.qpos[i];
    data->qvel[qvel_index[i]] = state.qvel[i];
    data->ctrl[actuator_ids[i]] = action.ctrl[i];
  }
}

void HumanoidEnv::step(const Action& action) {
  for (int i = 0; i < kNumJoints; ++i) {
    data->ctrl[actuator_ids[i]] = action.ctrl[i];
  }
  mj_step(model, data);
}

JointState HumanoidEnv::getState() const {
  JointState state{};
  for (int i = 0; i < kNumJoints; ++i) {
    state.qpos[i] = data->qpos[qpos_index[i]];
    state.qvel[i] = data->qvel[qvel_index[i]];
  }
  return state;
}

Action HumanoidEnv::getAction() const {
  Action action{};
  for (int i = 0; i < kNumJoints; ++i) {
    action.ctrl[i] = data->ctrl[actuator_ids[i]];
  }
  return action;
}

double HumanoidEnv::time() const {
  return data->time;
}

bool HumanoidEnv::stateIsFinite() const {
  const JointState state = getState();
  const Action action = getAction();
  if (!std::isfinite(data->time)) {
    return false;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    if (!std::isfinite(state.qpos[i]) || !std::isfinite(state.qvel[i]) ||
        !std::isfinite(action.ctrl[i])) {
      return false;
    }
  }
  return true;
}

bool validateAsset(const HumanoidEnv& env) {
  const mjModel* model = env.model;
  std::printf("Asset QA\n");

  const bool joints_ok = model->njnt == kNumJoints;
  const bool actuators_ok = model->nu == kNumJoints;
  const bool dofs_ok = model->nq == kNumJoints && model->nv == kNumJoints;
  const bool timestep_ok = std::fabs(model->opt.timestep - kExpectedTimestep) < 1e-12;

  printCheck("exactly 12 articulated joints", joints_ok);
  printCheck("exactly 12 actuators", actuators_ok);
  printCheck("joint DOFs consistent (nq=12, nv=12)", dofs_ok);
  printCheck("timestep = 0.002", timestep_ok);

  bool bodies_ok = true;
  for (int i = 0; i < kNumRequiredBodies; ++i) {
    const bool found = mj_name2id(model, mjOBJ_BODY, kRequiredBodies[i]) >= 0;
    printCheck(kRequiredBodies[i], found);
    bodies_ok = bodies_ok && found;
  }

  bool names_ok = true;
  for (int i = 0; i < kNumJoints; ++i) {
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, kJointNames[i]);
    const int actuator_id = mj_name2id(model, mjOBJ_ACTUATOR, kActuatorNames[i]);
    printCheck(kJointNames[i], joint_id >= 0);
    printCheck(kActuatorNames[i], actuator_id >= 0);
    names_ok = names_ok && joint_id >= 0 && actuator_id >= 0;
  }

  bool masses_ok = true;
  bool com_ok = true;
  bool inertia_ok = true;
  for (int i = 0; i < kNumMovingBodies; ++i) {
    const int body_id = mj_name2id(model, mjOBJ_BODY, kMovingBodies[i]);
    if (body_id < 0) {
      masses_ok = false;
      com_ok = false;
      inertia_ok = false;
      continue;
    }
    masses_ok = masses_ok && isFinitePositive(model->body_mass[body_id]);
    const double* com = model->body_ipos + 3 * body_id;
    com_ok = com_ok && std::isfinite(com[0]) && std::isfinite(com[1]) && std::isfinite(com[2]);
    const double* inertia = model->body_inertia + 3 * body_id;
    inertia_ok = inertia_ok && inertiaLooksPhysical(inertia[0], inertia[1], inertia[2]);
  }

  printCheck("moving-link masses finite and > 0", masses_ok);
  printCheck("centers of mass finite", com_ok);
  printCheck("inertia finite and physically valid", inertia_ok);

  return joints_ok && actuators_ok && dofs_ok && timestep_ok && bodies_ok && names_ok &&
         masses_ok && com_ok && inertia_ok;
}

}  // namespace simready
