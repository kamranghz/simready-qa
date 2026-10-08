#pragma once

#include <mujoco/mujoco.h>

namespace simready {

constexpr int kNumJoints = 12;
constexpr double kExpectedTimestep = 0.002;

extern const char* kJointNames[kNumJoints];
extern const char* kActuatorNames[kNumJoints];
extern const char* kRequiredBodies[];
extern const int kNumRequiredBodies;
extern const char* kMovingBodies[];
extern const int kNumMovingBodies;

struct JointState {
  double qpos[kNumJoints];
  double qvel[kNumJoints];
};

struct Action {
  double ctrl[kNumJoints];
};

// Compact policy-facing interface: reset, step, getState.
struct HumanoidEnv {
  mjModel* model = nullptr;
  mjData* data = nullptr;
  int qpos_index[kNumJoints]{};
  int qvel_index[kNumJoints]{};
  int actuator_ids[kNumJoints]{};

  bool load(const char* model_path, char* error, int error_size);
  void close();
  void reset(const JointState& state, const Action& action);
  void step(const Action& action);
  JointState getState() const;
  Action getAction() const;
  double time() const;
  bool stateIsFinite() const;
};

bool validateAsset(const HumanoidEnv& env);

}  // namespace simready
