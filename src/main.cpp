#include "simready/environment.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kReplayRuns = 3;
constexpr int kDefaultRollouts = 100;
constexpr int kDefaultSteps = 200;
constexpr std::uint32_t kDefaultSeed = 42;
constexpr double kReplayTolerance = 1e-12;
constexpr double kRegressionTolerance = 1e-9;
constexpr double kMaxAbsPosition = 6.283185307179586;  // 2*pi
constexpr double kMaxAbsVelocity = 40.0;

using simready::Action;
using simready::HumanoidEnv;
using simready::JointState;
using simready::kExpectedTimestep;
using simready::kJointNames;
using simready::kNumJoints;

struct CanonicalConfig {
  JointState state;
  Action action;
  int num_steps;
};

CanonicalConfig makeCanonicalConfig() {
  CanonicalConfig config{};
  const double qpos[kNumJoints] = {
      0.20, 0.15, 0.30, 0.20, -0.15, 0.30,
      0.10, 0.20, -0.05, 0.10, 0.20, -0.05};
  const double ctrl[kNumJoints] = {
      0.08, 0.04, 0.05, 0.08, -0.04, 0.05,
      0.04, 0.03, 0.02, 0.04, 0.03, 0.02};
  for (int i = 0; i < kNumJoints; ++i) {
    config.state.qpos[i] = qpos[i];
    config.state.qvel[i] = 0.0;
    config.action.ctrl[i] = ctrl[i];
  }
  config.num_steps = kDefaultSteps;
  return config;
}

double maxAbsDiff(const JointState& a, const JointState& b) {
  double max_diff = 0.0;
  for (int i = 0; i < kNumJoints; ++i) {
    max_diff = std::max(max_diff, std::fabs(a.qpos[i] - b.qpos[i]));
    max_diff = std::max(max_diff, std::fabs(a.qvel[i] - b.qvel[i]));
  }
  return max_diff;
}

bool statesMatch(const JointState& a, const JointState& b, double tolerance) {
  return maxAbsDiff(a, b) <= tolerance;
}

bool rolloutHealthy(const JointState& state, bool finite) {
  if (!finite) {
    return false;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    if (std::fabs(state.qpos[i]) > kMaxAbsPosition || std::fabs(state.qvel[i]) > kMaxAbsVelocity) {
      return false;
    }
  }
  return true;
}

double sampleUniform(std::mt19937& rng, double lo, double hi) {
  std::uniform_real_distribution<double> dist(lo, hi);
  return dist(rng);
}

void sampleRollout(std::mt19937& rng, JointState* state, Action* action) {
  for (int i = 0; i < kNumJoints; ++i) {
    state->qvel[i] = 0.0;
  }
  state->qpos[0] = sampleUniform(rng, -0.35, 0.55);
  state->qpos[1] = sampleUniform(rng, -0.05, 0.40);
  state->qpos[2] = sampleUniform(rng, 0.00, 0.70);
  state->qpos[3] = sampleUniform(rng, -0.35, 0.55);
  state->qpos[4] = sampleUniform(rng, -0.40, 0.05);
  state->qpos[5] = sampleUniform(rng, 0.00, 0.70);
  state->qpos[6] = sampleUniform(rng, -0.20, 0.40);
  state->qpos[7] = sampleUniform(rng, 0.00, 0.80);
  state->qpos[8] = sampleUniform(rng, -0.30, 0.30);
  state->qpos[9] = sampleUniform(rng, -0.20, 0.40);
  state->qpos[10] = sampleUniform(rng, 0.00, 0.80);
  state->qpos[11] = sampleUniform(rng, -0.30, 0.30);
  for (int i = 0; i < kNumJoints; ++i) {
    action->ctrl[i] = sampleUniform(rng, -0.08, 0.08);
  }
}

void printStateBlock(const char* title, const JointState& state) {
  std::printf("%s\n", title);
  for (int i = 0; i < kNumJoints; ++i) {
    std::printf("  %s: position = %.12f, velocity = %.12f\n",
                kJointNames[i], state.qpos[i], state.qvel[i]);
  }
}

std::string csvHeader() {
  std::ostringstream out;
  out << "rollout_id,seed,backend,step,time";
  for (int i = 0; i < kNumJoints; ++i) {
    out << ",qpos_" << kJointNames[i];
  }
  for (int i = 0; i < kNumJoints; ++i) {
    out << ",qvel_" << kJointNames[i];
  }
  for (int i = 0; i < kNumJoints; ++i) {
    out << ",ctrl_" << kJointNames[i];
  }
  return out.str();
}

void writeCsvRow(std::ofstream& file, int rollout_id, std::uint32_t seed, int step, double time,
                 const JointState& state, const Action& action) {
  file << rollout_id << ',' << seed << ",cpu," << step << ',' << std::setprecision(17) << time;
  for (int i = 0; i < kNumJoints; ++i) {
    file << ',' << state.qpos[i];
  }
  for (int i = 0; i < kNumJoints; ++i) {
    file << ',' << state.qvel[i];
  }
  for (int i = 0; i < kNumJoints; ++i) {
    file << ',' << action.ctrl[i];
  }
  file << '\n';
}

bool runDeterministicReplay(HumanoidEnv* env, const CanonicalConfig& config) {
  std::printf("Deterministic replay (3 identical resets)\n");
  JointState results[kReplayRuns];
  for (int run = 0; run < kReplayRuns; ++run) {
    env->reset(config.state, config.action);
    for (int step = 0; step < config.num_steps; ++step) {
      env->step(config.action);
    }
    results[run] = env->getState();
    char title[32];
    std::snprintf(title, sizeof(title), "Run %d", run + 1);
    printStateBlock(title, results[run]);
  }
  bool pass = true;
  for (int run = 1; run < kReplayRuns; ++run) {
    if (!statesMatch(results[0], results[run], kReplayTolerance)) {
      pass = false;
    }
  }
  std::printf("Deterministic replay: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

bool runRegression(HumanoidEnv* env, const CanonicalConfig& config, const char* baseline_path) {
  std::printf("Regression\n");
  env->reset(config.state, config.action);
  for (int step = 0; step < config.num_steps; ++step) {
    env->step(config.action);
  }
  const JointState actual = env->getState();

  std::ifstream in(baseline_path);
  if (!in) {
    std::printf("  could not read baseline: %s\n", baseline_path);
    std::printf("Regression: FAIL\n\n");
    return false;
  }

  JointState expected{};
  std::string key;
  while (in >> key) {
    if (key == "final_qpos") {
      for (int i = 0; i < kNumJoints; ++i) {
        in >> expected.qpos[i];
      }
    } else if (key == "final_qvel") {
      for (int i = 0; i < kNumJoints; ++i) {
        in >> expected.qvel[i];
      }
    } else {
      std::string skip;
      std::getline(in, skip);
    }
  }

  const double max_diff = maxAbsDiff(actual, expected);
  const bool pass = max_diff <= kRegressionTolerance;
  std::printf("  max abs state difference = %.12e\n", max_diff);
  std::printf("  tolerance = %.12e\n", kRegressionTolerance);
  std::printf("Regression: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

void runThroughput(HumanoidEnv* env, const CanonicalConfig& config, int num_rollouts) {
  using clock = std::chrono::steady_clock;
  const int total_steps = num_rollouts * config.num_steps;
  const auto t0 = clock::now();
  for (int n = 0; n < num_rollouts; ++n) {
    env->reset(config.state, config.action);
    for (int step = 0; step < config.num_steps; ++step) {
      env->step(config.action);
    }
  }
  const auto t1 = clock::now();
  const double seconds = std::chrono::duration<double>(t1 - t0).count();
  const double steps_per_sec = total_steps / seconds;
  const double rollouts_per_sec = num_rollouts / seconds;

  std::printf("Throughput (CPU, in-memory, no file I/O)\n");
  std::printf("  rollouts = %d\n", num_rollouts);
  std::printf("  steps per rollout = %d\n", config.num_steps);
  std::printf("  total simulated steps = %d\n", total_steps);
  std::printf("  wall-clock = %.6f s\n", seconds);
  std::printf("  simulation steps/s = %.1f\n", steps_per_sec);
  std::printf("  rollouts/s = %.1f\n\n", rollouts_per_sec);
}

struct DatasetStats {
  int success = 0;
  int failure = 0;
  bool replay_from_seed_pass = false;
};

DatasetStats generateDataset(HumanoidEnv* env, const std::filesystem::path& out_dir,
                             std::uint32_t seed, int num_rollouts, int num_steps) {
  DatasetStats stats;
  std::filesystem::create_directories(out_dir);
  const auto steps_path = out_dir / "rollouts.csv";
  const auto summary_path = out_dir / "rollout_summary.csv";

  std::ofstream steps_file(steps_path);
  std::ofstream summary_file(summary_path);
  steps_file << csvHeader() << '\n';
  summary_file << "rollout_id,seed,label,final_time,max_abs_qpos,max_abs_qvel\n";
  steps_file << std::setprecision(17);
  summary_file << std::setprecision(17);

  std::mt19937 rng(seed);
  JointState first_state{};
  Action first_action{};

  for (int id = 0; id < num_rollouts; ++id) {
    JointState init{};
    Action action{};
    sampleRollout(rng, &init, &action);
    if (id == 0) {
      first_state = init;
      first_action = action;
    }

    env->reset(init, action);
    bool healthy = rolloutHealthy(env->getState(), env->stateIsFinite());
    double max_abs_qpos = 0.0;
    double max_abs_qvel = 0.0;

    for (int step = 0; step < num_steps; ++step) {
      env->step(action);
      const JointState state = env->getState();
      healthy = healthy && rolloutHealthy(state, env->stateIsFinite());
      writeCsvRow(steps_file, id, seed, step + 1, env->time(), state, action);
      for (int i = 0; i < kNumJoints; ++i) {
        max_abs_qpos = std::max(max_abs_qpos, std::fabs(state.qpos[i]));
        max_abs_qvel = std::max(max_abs_qvel, std::fabs(state.qvel[i]));
      }
    }

    const bool time_ok = std::fabs(env->time() - num_steps * kExpectedTimestep) < 1e-9;
    healthy = healthy && time_ok;
    if (healthy) {
      ++stats.success;
    } else {
      ++stats.failure;
    }
    summary_file << id << ',' << seed << ',' << (healthy ? "SUCCESS" : "FAILURE") << ','
                 << env->time() << ',' << max_abs_qpos << ',' << max_abs_qvel << '\n';
  }

  std::printf("Synthetic rollouts\n");
  std::printf("  seed = %u\n", seed);
  std::printf("  rollouts = %d, steps each = %d\n", num_rollouts, num_steps);
  std::printf("  wrote %s\n", steps_path.string().c_str());
  std::printf("  wrote %s\n", summary_path.string().c_str());
  std::printf("  SUCCESS = %d\n", stats.success);
  std::printf("  FAILURE = %d\n", stats.failure);

  // Reproduce rollout 0 from the same seed and first sampled configuration.
  env->reset(first_state, first_action);
  std::ifstream check(steps_path);
  std::string header;
  std::getline(check, header);
  bool replay_ok = true;
  int matched_rows = 0;
  for (int step = 0; step < num_steps; ++step) {
    env->step(first_action);
    const JointState state = env->getState();
    std::string line;
    std::getline(check, line);
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    int col = 0;
    while (std::getline(ss, cell, ',')) {
      if (col >= 5) {
        values.push_back(std::stod(cell));
      }
      ++col;
    }
    if (static_cast<int>(values.size()) < 2 * kNumJoints) {
      replay_ok = false;
      break;
    }
    JointState logged{};
    for (int i = 0; i < kNumJoints; ++i) {
      logged.qpos[i] = values[i];
      logged.qvel[i] = values[kNumJoints + i];
    }
    if (!statesMatch(state, logged, kReplayTolerance)) {
      replay_ok = false;
      break;
    }
    ++matched_rows;
  }
  stats.replay_from_seed_pass = replay_ok && matched_rows == num_steps;
  std::printf("Replay from seed (rollout 0): %s\n\n",
              stats.replay_from_seed_pass ? "PASS" : "FAIL");
  return stats;
}

struct TraceRow {
  int step = 0;
  double time = 0.0;
  JointState state{};
};

void writeReferenceTrace(const std::filesystem::path& path, const std::vector<TraceRow>& rows) {
  std::ofstream file(path);
  file << "step,time";
  for (int i = 0; i < kNumJoints; ++i) {
    file << ",qpos_" << kJointNames[i];
  }
  for (int i = 0; i < kNumJoints; ++i) {
    file << ",qvel_" << kJointNames[i];
  }
  file << '\n' << std::setprecision(17);
  for (const TraceRow& row : rows) {
    file << row.step << ',' << row.time;
    for (int i = 0; i < kNumJoints; ++i) {
      file << ',' << row.state.qpos[i];
    }
    for (int i = 0; i < kNumJoints; ++i) {
      file << ',' << row.state.qvel[i];
    }
    file << '\n';
  }
}

std::vector<TraceRow> readReferenceTrace(const std::filesystem::path& path) {
  std::vector<TraceRow> rows;
  std::ifstream in(path);
  std::string header;
  std::getline(in, header);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    TraceRow row;
    std::getline(ss, cell, ',');
    row.step = std::stoi(cell);
    std::getline(ss, cell, ',');
    row.time = std::stod(cell);
    for (int i = 0; i < kNumJoints; ++i) {
      std::getline(ss, cell, ',');
      row.state.qpos[i] = std::stod(cell);
    }
    for (int i = 0; i < kNumJoints; ++i) {
      std::getline(ss, cell, ',');
      row.state.qvel[i] = std::stod(cell);
    }
    rows.push_back(row);
  }
  return rows;
}

std::vector<TraceRow> simulateTrace(HumanoidEnv* env, const CanonicalConfig& config) {
  std::vector<TraceRow> rows;
  env->reset(config.state, config.action);
  for (int step = 0; step < config.num_steps; ++step) {
    env->step(config.action);
    TraceRow row;
    row.step = step + 1;
    row.time = env->time();
    row.state = env->getState();
    rows.push_back(row);
  }
  return rows;
}

bool runFidelity(HumanoidEnv* env, const CanonicalConfig& config,
                 const std::filesystem::path& reference_path,
                 const std::filesystem::path& written_path) {
  const std::vector<TraceRow> simulated = simulateTrace(env, config);
  writeReferenceTrace(written_path, simulated);

  std::printf("Reference-trace fidelity\n");
  std::printf("  reference file = %s\n", reference_path.string().c_str());
  std::printf("  reference type = synthetic example trace (not hardware)\n");

  if (!std::filesystem::exists(reference_path)) {
    std::printf("  committed reference not found; wrote %s for software validation\n",
                written_path.string().c_str());
    std::printf("Reference-trace fidelity: FAIL\n\n");
    return false;
  }

  const std::vector<TraceRow> reference = readReferenceTrace(reference_path);
  if (reference.size() != simulated.size() || reference.empty()) {
    std::printf("  length mismatch: reference=%zu simulated=%zu\n",
                reference.size(), simulated.size());
    std::printf("Reference-trace fidelity: FAIL\n\n");
    return false;
  }

  double sse_all = 0.0;
  double sse_joint[kNumJoints] = {};
  double sse_vel_all = 0.0;
  const int n = static_cast<int>(simulated.size());
  for (int t = 0; t < n; ++t) {
    for (int i = 0; i < kNumJoints; ++i) {
      const double dp = simulated[t].state.qpos[i] - reference[t].state.qpos[i];
      const double dv = simulated[t].state.qvel[i] - reference[t].state.qvel[i];
      sse_joint[i] += dp * dp;
      sse_all += dp * dp;
      sse_vel_all += dv * dv;
    }
  }

  std::printf("  per-joint position RMSE:\n");
  for (int i = 0; i < kNumJoints; ++i) {
    const double rmse = std::sqrt(sse_joint[i] / n);
    std::printf("    %s = %.12e\n", kJointNames[i], rmse);
  }
  const double pos_rmse = std::sqrt(sse_all / (n * kNumJoints));
  const double vel_rmse = std::sqrt(sse_vel_all / (n * kNumJoints));
  std::printf("  overall position RMSE = %.12e\n", pos_rmse);
  std::printf("  overall velocity RMSE = %.12e\n", vel_rmse);
  const bool pass = pos_rmse <= 1e-9;
  std::printf("Reference-trace fidelity: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = DEFAULT_MODEL_PATH;
  if (argc > 1) {
    model_path = argv[1];
  }

  std::printf("=== SimReadyQA ===\n");
  std::printf("MuJoCo version: %s\n", mj_versionString());
  std::printf("Loading MJCF: %s\n\n", model_path);

  HumanoidEnv env;
  char load_error[1024] = "unknown error";
  if (!env.load(model_path, load_error, sizeof(load_error))) {
    std::printf("Failed to load model: %s\n", load_error);
    std::printf("ASSET QA: FAIL\n");
    return 1;
  }

  std::printf("Model loaded successfully.\n");
  std::printf("  nbody = %d\n", static_cast<int>(env.model->nbody));
  std::printf("  njnt  = %d\n", static_cast<int>(env.model->njnt));
  std::printf("  nq    = %d\n", static_cast<int>(env.model->nq));
  std::printf("  nv    = %d\n", static_cast<int>(env.model->nv));
  std::printf("  nu    = %d\n", static_cast<int>(env.model->nu));
  std::printf("  timestep = %.6f s\n\n", env.model->opt.timestep);

  const bool qa_pass = simready::validateAsset(env);
  std::printf("ASSET QA: %s\n\n", qa_pass ? "PASS" : "FAIL");
  if (!qa_pass) {
    env.close();
    return 1;
  }

  const CanonicalConfig canonical = makeCanonicalConfig();
  std::printf("Canonical config: fixed pose (no RNG)\n");
  std::printf("  steps = %d, timestep = %.6f s\n\n", canonical.num_steps, kExpectedTimestep);

  const bool replay_pass = runDeterministicReplay(&env, canonical);

  {
    std::filesystem::create_directories(OUTPUTS_DIR);
    const JointState final_state = env.getState();
    std::ofstream dump((std::filesystem::path(OUTPUTS_DIR) / "cpu_canonical_final.txt").string());
    dump << std::setprecision(17) << "qpos";
    for (int i = 0; i < kNumJoints; ++i) {
      dump << ' ' << final_state.qpos[i];
    }
    dump << "\nqvel";
    for (int i = 0; i < kNumJoints; ++i) {
      dump << ' ' << final_state.qvel[i];
    }
    dump << '\n';
  }

  const bool regression_pass =
      runRegression(&env, canonical, (std::filesystem::path(ASSETS_DIR) / "regression_baseline.txt").string().c_str());
  runThroughput(&env, canonical, kDefaultRollouts);

  const auto outputs = std::filesystem::path(OUTPUTS_DIR);
  const DatasetStats dataset =
      generateDataset(&env, outputs, kDefaultSeed, kDefaultRollouts, kDefaultSteps);

  const auto written_trace = outputs / "example_reference_trace.csv";
  const auto committed_trace = std::filesystem::path(ASSETS_DIR) / "example_reference_trace.csv";
  const bool fidelity_pass = runFidelity(&env, canonical, committed_trace, written_trace);

  env.close();

  const bool pass = replay_pass && regression_pass && dataset.replay_from_seed_pass && fidelity_pass;
  std::printf("Harness summary: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
