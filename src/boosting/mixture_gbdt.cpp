/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Mixture-of-Experts GBDT extension for regime-switching models.
 */
#include "mixture_gbdt.h"

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace LightGBM {

constexpr double kMixtureEpsilon = 1e-12;

MixtureGBDT::MixtureGBDT()
    : num_experts_(4),
      train_data_(nullptr),
      objective_function_(nullptr),
      num_data_(0),
      iter_(0),
      max_feature_idx_(0),
      label_idx_(0),
      use_markov_(false),
      use_momentum_(false),
      early_stopping_round_(0),
      early_stopping_min_delta_(0.0),
      es_first_metric_only_(false) {
}

MixtureGBDT::~MixtureGBDT() {
}

void MixtureGBDT::Init(const Config* config, const Dataset* train_data,
                       const ObjectiveFunction* objective_function,
                       const std::vector<const Metric*>& training_metrics) {
  CHECK_NOTNULL(train_data);
  train_data_ = train_data;
  objective_function_ = objective_function;
  training_metrics_ = training_metrics;
  num_data_ = train_data_->num_data();
  iter_ = 0;

  // Store original config
  config_ = std::unique_ptr<Config>(new Config(*config));
  num_experts_ = config_->mixture_num_experts;

  // Get feature info
  max_feature_idx_ = train_data_->num_total_features() - 1;
  label_idx_ = train_data_->label_idx();
  feature_names_ = train_data_->feature_names();

  // Determine E-step loss type
  if (config_->mixture_e_step_loss == "auto") {
    // Infer from objective
    if (config_->objective == "regression_l1" || config_->objective == "l1" ||
        config_->objective == "mean_absolute_error" || config_->objective == "mae") {
      e_step_loss_type_ = "l1";
    } else if (config_->objective == "quantile") {
      e_step_loss_type_ = "quantile";
    } else {
      e_step_loss_type_ = "l2";  // default fallback
    }
  } else {
    e_step_loss_type_ = config_->mixture_e_step_loss;
  }

  Log::Info("MixtureGBDT: Initializing with %d experts, E-step loss type: %s",
            num_experts_, e_step_loss_type_.c_str());

  // Create expert config (same as original but for regression)
  expert_config_ = std::unique_ptr<Config>(new Config(*config));
  // Experts use the same objective as the mixture

  // Create gate config (multiclass classification)
  gate_config_ = std::unique_ptr<Config>(new Config(*config));
  gate_config_->objective = "multiclass";
  gate_config_->num_class = num_experts_;
  gate_config_->max_depth = config_->mixture_gate_max_depth;
  gate_config_->num_leaves = config_->mixture_gate_num_leaves;
  gate_config_->learning_rate = config_->mixture_gate_learning_rate;
  gate_config_->lambda_l2 = config_->mixture_gate_lambda_l2;

  // Initialize experts
  // Note: We pass nullptr for objective_function because we use custom gradients
  // (responsibility-weighted) in MStepExperts. The main objective is stored in
  // objective_function_ and used to compute gradients on yhat.
  // Each expert gets a different seed to break symmetry when using uniform initialization.
  // Per-expert hyperparameters (max_depth, num_leaves, min_data_in_leaf, min_gain_to_split) can be specified.
  Log::Debug("MixtureGBDT::Init - creating %d experts", num_experts_);

  // Validate per-expert hyperparameters if provided
  const bool use_per_expert_max_depth = !config_->mixture_expert_max_depths.empty();
  const bool use_per_expert_num_leaves = !config_->mixture_expert_num_leaves.empty();
  const bool use_per_expert_min_data_in_leaf = !config_->mixture_expert_min_data_in_leaf.empty();
  const bool use_per_expert_min_gain_to_split = !config_->mixture_expert_min_gain_to_split.empty();

  if (use_per_expert_max_depth &&
      static_cast<int>(config_->mixture_expert_max_depths.size()) != num_experts_) {
    Log::Fatal("mixture_expert_max_depths must have exactly %d values (one per expert), got %d",
               num_experts_, static_cast<int>(config_->mixture_expert_max_depths.size()));
  }
  if (use_per_expert_num_leaves &&
      static_cast<int>(config_->mixture_expert_num_leaves.size()) != num_experts_) {
    Log::Fatal("mixture_expert_num_leaves must have exactly %d values (one per expert), got %d",
               num_experts_, static_cast<int>(config_->mixture_expert_num_leaves.size()));
  }
  if (use_per_expert_min_data_in_leaf &&
      static_cast<int>(config_->mixture_expert_min_data_in_leaf.size()) != num_experts_) {
    Log::Fatal("mixture_expert_min_data_in_leaf must have exactly %d values (one per expert), got %d",
               num_experts_, static_cast<int>(config_->mixture_expert_min_data_in_leaf.size()));
  }
  if (use_per_expert_min_gain_to_split &&
      static_cast<int>(config_->mixture_expert_min_gain_to_split.size()) != num_experts_) {
    Log::Fatal("mixture_expert_min_gain_to_split must have exactly %d values (one per expert), got %d",
               num_experts_, static_cast<int>(config_->mixture_expert_min_gain_to_split.size()));
  }

  experts_.clear();
  experts_.reserve(num_experts_);
  expert_configs_.clear();
  expert_configs_.reserve(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    Log::Debug("MixtureGBDT::Init - creating expert %d", k);
    // Create per-expert config with different seed for symmetry breaking
    expert_configs_.emplace_back(new Config(*expert_config_));
    expert_configs_[k]->seed = config_->seed + k + 1;  // Different seed per expert

    // Apply per-expert hyperparameters if specified
    if (use_per_expert_max_depth) {
      expert_configs_[k]->max_depth = config_->mixture_expert_max_depths[k];
    }
    if (use_per_expert_num_leaves) {
      expert_configs_[k]->num_leaves = config_->mixture_expert_num_leaves[k];
    }
    if (use_per_expert_min_data_in_leaf) {
      expert_configs_[k]->min_data_in_leaf = config_->mixture_expert_min_data_in_leaf[k];
    }
    if (use_per_expert_min_gain_to_split) {
      expert_configs_[k]->min_gain_to_split = config_->mixture_expert_min_gain_to_split[k];
    }

    experts_.emplace_back(new GBDT());
    Log::Debug("MixtureGBDT::Init - initializing expert %d with seed %d, max_depth=%d, num_leaves=%d, min_data=%d, min_gain=%.4f",
               k, expert_configs_[k]->seed, expert_configs_[k]->max_depth,
               expert_configs_[k]->num_leaves, expert_configs_[k]->min_data_in_leaf,
               expert_configs_[k]->min_gain_to_split);
    experts_[k]->Init(expert_configs_[k].get(), train_data_, nullptr, {});
    Log::Debug("MixtureGBDT::Init - expert %d initialized", k);
  }

  // Check smoothing modes
  use_markov_ = (config_->mixture_r_smoothing == "markov");
  use_momentum_ = (config_->mixture_r_smoothing == "momentum");

  // Initialize gate
  Log::Debug("MixtureGBDT::Init - creating gate");
  gate_.reset(new GBDT());
  Log::Debug("MixtureGBDT::Init - initializing gate");

  if (use_markov_) {
    Log::Info("MixtureGBDT: Markov mode enabled (lambda=%.2f)",
              config_->mixture_smoothing_lambda);
  } else if (use_momentum_) {
    Log::Info("MixtureGBDT: Momentum mode enabled (lambda=%.2f)",
              config_->mixture_smoothing_lambda);
  }
  gate_->Init(gate_config_.get(), train_data_, nullptr, {});
  Log::Debug("MixtureGBDT::Init - gate initialized");

  // Allocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  gate_proba_.resize(nk);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  // Initialize Markov-specific buffers
  if (use_markov_) {
    prev_gate_proba_.resize(nk);
    const double uniform_prob = 1.0 / num_experts_;
    std::fill(prev_gate_proba_.begin(), prev_gate_proba_.end(), uniform_prob);
  }

  // Initialize expert bias for loss-free load balancing
  expert_bias_.resize(num_experts_, 0.0);

  // Initialize expert dropout RNG
  dropout_rng_.seed(config_->seed + 12345);  // Different seed from other RNGs
  dropout_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);

  // Initialize early stopping parameters
  early_stopping_round_ = config_->early_stopping_round;
  early_stopping_min_delta_ = config_->early_stopping_min_delta;
  es_first_metric_only_ = config_->first_metric_only;

  // Initialize responsibilities
  InitResponsibilities();

  Log::Info("MixtureGBDT: Initialization complete (smoothing=%s)",
            config_->mixture_r_smoothing.c_str());
}

void MixtureGBDT::InitResponsibilities() {
  const label_t* labels = train_data_->metadata().label();

  if (config_->mixture_init == "quantile") {
    // Quantile-based initialization: assign samples to experts based on label quantiles
    // This breaks symmetry by giving each expert a different subset of data
    Log::Info("MixtureGBDT: Using quantile-based initialization");

    // Sort indices by label value
    std::vector<data_size_t> sorted_indices(num_data_);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [labels](data_size_t a, data_size_t b) { return labels[a] < labels[b]; });

    // Assign to experts based on quantiles with soft boundaries
    const double base_r = 0.1 / num_experts_;  // Small base probability for all experts
    const double main_r = 1.0 - base_r * num_experts_;  // Main probability for assigned expert

    for (data_size_t rank = 0; rank < num_data_; ++rank) {
      data_size_t i = sorted_indices[rank];
      int assigned_expert = static_cast<int>(rank * num_experts_ / num_data_);
      if (assigned_expert >= num_experts_) assigned_expert = num_experts_ - 1;

      for (int k = 0; k < num_experts_; ++k) {
        if (k == assigned_expert) {
          responsibilities_[i * num_experts_ + k] = main_r + base_r;
        } else {
          responsibilities_[i * num_experts_ + k] = base_r;
        }
      }
    }
  } else if (config_->mixture_init == "random") {
    // Random initialization: randomly assign samples to experts
    Log::Info("MixtureGBDT: Using random initialization");

    std::mt19937 rng(config_->seed);
    std::uniform_int_distribution<int> dist(0, num_experts_ - 1);

    const double base_r = 0.1 / num_experts_;
    const double main_r = 1.0 - base_r * num_experts_;

    for (data_size_t i = 0; i < num_data_; ++i) {
      int assigned_expert = dist(rng);
      for (int k = 0; k < num_experts_; ++k) {
        if (k == assigned_expert) {
          responsibilities_[i * num_experts_ + k] = main_r + base_r;
        } else {
          responsibilities_[i * num_experts_ + k] = base_r;
        }
      }
    }
  } else {
    // Default: uniform initialization
    // All experts start with equal responsibility (1/K).
    // Symmetry is broken by per-expert seeds (set in Init()).
    // This allows experts to naturally specialize based on prediction error,
    // without explicit label-based assignment.
    Log::Info("MixtureGBDT: Using uniform initialization (symmetry broken by per-expert seeds)");

    const double uniform_r = 1.0 / num_experts_;
    for (data_size_t i = 0; i < num_data_; ++i) {
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] = uniform_r;
      }
    }
  }
}

void MixtureGBDT::Softmax(const double* scores, int n, double* probs) const {
  // Find max for numerical stability
  double max_score = scores[0];
  for (int i = 1; i < n; ++i) {
    if (scores[i] > max_score) max_score = scores[i];
  }

  // Compute exp and sum
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    probs[i] = std::exp(scores[i] - max_score);
    sum += probs[i];
  }

  // Normalize
  for (int i = 0; i < n; ++i) {
    probs[i] /= sum;
  }
}

double MixtureGBDT::ComputePointwiseLoss(double y, double pred) const {
  double diff = y - pred;
  if (e_step_loss_type_ == "l2") {
    return diff * diff;
  } else if (e_step_loss_type_ == "l1") {
    return std::fabs(diff);
  } else if (e_step_loss_type_ == "quantile") {
    // TODO(shiyu1994): Get quantile alpha from config
    double alpha = 0.5;  // default median
    if (diff >= 0) {
      return alpha * diff;
    } else {
      return (alpha - 1.0) * diff;
    }
  }
  // Default to L2
  return diff * diff;
}

void MixtureGBDT::Forward() {
  // Get expert predictions
  for (int k = 0; k < num_experts_; ++k) {
    int64_t out_len;
    experts_[k]->GetPredictAt(0, expert_pred_.data() + k * num_data_, &out_len);
  }

  // Get gate probabilities (softmax of gate raw predictions)
  // Note: GetPredictAt returns class-major order (all class 0, then class 1, etc.)
  std::vector<double> gate_raw(static_cast<size_t>(num_data_) * num_experts_);
  int64_t out_len;
  gate_->GetPredictAt(0, gate_raw.data(), &out_len);

  // Apply softmax per sample with expert bias for load balancing
  // gate_raw is in class-major order: gate_raw[k * num_data_ + i] = score for sample i, class k
  // gate_proba_ is in sample-major order: gate_proba_[i * num_experts_ + k]
  // expert_bias_ is added to encourage balanced expert usage (Loss-Free Balancing)
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    // Copy to sample-major order for this sample, adding expert bias
    std::vector<double> scores(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      scores[k] = gate_raw[k * num_data_ + i] + expert_bias_[k];  // Add bias for load balancing
    }
    Softmax(scores.data(), num_experts_,
            gate_proba_.data() + i * num_experts_);
  }

  // Markov mode: blend gate_proba with prev_gate_proba
  // This makes regime transitions smoother and dependent on previous state
  if (use_markov_) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double sum = 0.0;
        for (int k = 0; k < num_experts_; ++k) {
          size_t idx = i * num_experts_ + k;
          // Blend: new_proba = (1-lambda) * current + lambda * prev
          gate_proba_[idx] = (1.0 - lambda) * gate_proba_[idx] +
                             lambda * prev_gate_proba_[idx];
          sum += gate_proba_[idx];
        }
        // Renormalize (should be close to 1 already, but for numerical stability)
        for (int k = 0; k < num_experts_; ++k) {
          gate_proba_[i * num_experts_ + k] /= sum;
        }
      }
    }

    // Update prev_gate_proba with current values (for next iteration)
    // Using row-wise copy: prev[i] = current[i-1] for time series
    // First row keeps its initial/previous value
    for (data_size_t i = num_data_ - 1; i > 0; --i) {
      for (int k = 0; k < num_experts_; ++k) {
        prev_gate_proba_[i * num_experts_ + k] = gate_proba_[(i - 1) * num_experts_ + k];
      }
    }
    // First row: use current gate_proba (no previous available in this batch)
    // This maintains consistency for the first sample
  }

  // Compute combined prediction: yhat[i] = sum_k gate_proba[i,k] * expert_pred[i,k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      sum += gate_proba_[i * num_experts_ + k] * expert_pred_[k * num_data_ + i];
    }
    yhat_[i] = sum;
  }
}

void MixtureGBDT::ForwardValid(int valid_idx) {
  CHECK(valid_idx >= 0 && valid_idx < static_cast<int>(valid_datas_.size()));

  const Dataset* valid_data = valid_datas_[valid_idx];
  data_size_t num_valid = valid_data->num_data();
  int data_idx = valid_idx + 1;  // data_idx 0 is training data

  std::vector<double>& expert_pred = expert_pred_valid_[valid_idx];
  std::vector<double>& gate_proba = gate_proba_valid_[valid_idx];
  std::vector<double>& yhat = yhat_valid_[valid_idx];

  // Get expert predictions on validation data
  for (int k = 0; k < num_experts_; ++k) {
    int64_t out_len;
    experts_[k]->GetPredictAt(data_idx, expert_pred.data() + k * num_valid, &out_len);
  }

  // Get gate raw predictions on validation data (class-major order)
  std::vector<double> gate_raw(static_cast<size_t>(num_valid) * num_experts_);
  int64_t out_len;
  gate_->GetPredictAt(data_idx, gate_raw.data(), &out_len);

  // Apply softmax per sample with expert bias
  // gate_raw is in class-major order: gate_raw[k * num_valid + i]
  // gate_proba is in sample-major order: gate_proba[i * num_experts_ + k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_valid; ++i) {
    std::vector<double> scores(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      scores[k] = gate_raw[k * num_valid + i] + expert_bias_[k];
    }
    Softmax(scores.data(), num_experts_, gate_proba.data() + i * num_experts_);
  }

  // Markov mode: blend gate_proba with prev_gate_proba
  if (use_markov_ && valid_idx < static_cast<int>(prev_gate_proba_valid_.size())) {
    const double lambda = config_->mixture_smoothing_lambda;
    std::vector<double>& prev_gate_proba = prev_gate_proba_valid_[valid_idx];
    if (lambda > 0.0) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_valid; ++i) {
        double sum = 0.0;
        for (int k = 0; k < num_experts_; ++k) {
          size_t idx = i * num_experts_ + k;
          gate_proba[idx] = (1.0 - lambda) * gate_proba[idx] + lambda * prev_gate_proba[idx];
          sum += gate_proba[idx];
        }
        // Renormalize
        for (int k = 0; k < num_experts_; ++k) {
          gate_proba[i * num_experts_ + k] /= sum;
        }
      }
    }

    // Update prev_gate_proba for next iteration (time series shift)
    for (data_size_t i = num_valid - 1; i > 0; --i) {
      for (int k = 0; k < num_experts_; ++k) {
        prev_gate_proba[i * num_experts_ + k] = gate_proba[(i - 1) * num_experts_ + k];
      }
    }
  }

  // Compute combined prediction: yhat[i] = sum_k gate_proba[i,k] * expert_pred[k][i]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_valid; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      sum += gate_proba[i * num_experts_ + k] * expert_pred[k * num_valid + i];
    }
    yhat[i] = sum;
  }
}

void MixtureGBDT::EStep() {
  const label_t* labels = train_data_->metadata().label();
  const double alpha = config_->mixture_e_step_alpha;
  const bool use_loss_only = (config_->mixture_e_step_mode == "loss_only");

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    // Create local scores vector for each thread
    std::vector<double> scores(num_experts_);

    // Compute scores based on mode
    for (int k = 0; k < num_experts_; ++k) {
      double expert_p = expert_pred_[k * num_data_ + i];
      double loss = ComputePointwiseLoss(labels[i], expert_p);

      if (use_loss_only) {
        // loss_only mode: use only expert loss, ignore gate probability
        // s_ik = -alpha * loss(y_i, expert_pred_ik)
        // Lower loss → higher score → higher responsibility
        scores[k] = -alpha * loss;
      } else {
        // em mode (default): use both gate probability and expert loss
        // s_ik = log(gate_proba_ik + eps) - alpha * loss(y_i, expert_pred_ik)
        double gate_prob = gate_proba_[i * num_experts_ + k];
        scores[k] = std::log(gate_prob + kMixtureEpsilon) - alpha * loss;
      }
    }

    // Apply softmax to get responsibilities
    // Note: No hard clipping here. Collapse prevention is handled by
    // Loss-Free Load Balancing (UpdateExpertBias) which adjusts gate bias.
    Softmax(scores.data(), num_experts_, responsibilities_.data() + i * num_experts_);
  }
}

void MixtureGBDT::SmoothResponsibilities() {
  const double lambda = config_->mixture_smoothing_lambda;
  if (lambda <= 0.0) {
    return;
  }

  if (config_->mixture_r_smoothing == "ema") {
    // Apply EMA in row order (assumed to be time order)
    // r[i] = (1-lambda)*r[i] + lambda*r[i-1]
    for (data_size_t i = 1; i < num_data_; ++i) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        size_t idx = i * num_experts_ + k;
        size_t prev_idx = (i - 1) * num_experts_ + k;
        responsibilities_[idx] = (1.0 - lambda) * responsibilities_[idx] +
                                 lambda * responsibilities_[prev_idx];
        sum += responsibilities_[idx];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] /= sum;
      }
    }
  } else if (config_->mixture_r_smoothing == "momentum") {
    // Momentum smoothing: EMA with trend (direction of change)
    // extrapolated[i] = r[i-1] + lambda * (r[i-1] - r[i-2])
    // r_smooth[i] = (1-lambda)*r[i] + lambda*extrapolated[i]
    // This captures "inertia" - if regime is trending in a direction, continue that trend

    for (data_size_t i = 1; i < num_data_; ++i) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        size_t idx = i * num_experts_ + k;
        size_t prev_idx = (i - 1) * num_experts_ + k;

        double extrapolated;
        if (i >= 2) {
          // Use trend from previous samples
          size_t prev2_idx = (i - 2) * num_experts_ + k;
          double trend = responsibilities_[prev_idx] - responsibilities_[prev2_idx];
          extrapolated = responsibilities_[prev_idx] + lambda * trend;
        } else {
          // Not enough history, just use previous value
          extrapolated = responsibilities_[prev_idx];
        }

        // Blend current with extrapolated
        responsibilities_[idx] = (1.0 - lambda) * responsibilities_[idx] +
                                 lambda * extrapolated;
        // Clip to valid range
        if (responsibilities_[idx] < 0.0) responsibilities_[idx] = 0.0;
        if (responsibilities_[idx] > 1.0) responsibilities_[idx] = 1.0;
        sum += responsibilities_[idx];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] /= (sum + kMixtureEpsilon);
      }
    }
  }
}

void MixtureGBDT::UpdateExpertBias() {
  // Loss-Free Load Balancing: adjust expert bias when usage falls below threshold
  // Reference: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts" (2024)
  //
  // Modified for regime-switching: only intervene when an expert's usage
  // falls below the minimum threshold, allowing natural imbalanced distributions.
  //
  // min_usage = 1 / (balance_factor * K)
  // e.g., factor=10, K=2 -> min_usage = 5%, allows 95:5 imbalance

  const double min_usage = 1.0 / (config_->mixture_balance_factor * num_experts_);
  const double bias_update_rate = 0.1;  // η: how quickly to adjust bias

  // Compute actual load per expert (mean responsibility)
  std::vector<double> actual_load(num_experts_, 0.0);
  for (data_size_t i = 0; i < num_data_; ++i) {
    for (int k = 0; k < num_experts_; ++k) {
      actual_load[k] += responsibilities_[i * num_experts_ + k];
    }
  }
  for (int k = 0; k < num_experts_; ++k) {
    actual_load[k] /= num_data_;  // Normalize to [0, 1]
  }

  // Update bias: only increase for underloaded experts (below threshold)
  // Do NOT decrease for overloaded - allow natural imbalance
  for (int k = 0; k < num_experts_; ++k) {
    if (actual_load[k] < min_usage) {
      double load_diff = min_usage - actual_load[k];
      expert_bias_[k] += bias_update_rate * load_diff;
    }
    // Note: We don't decrease bias for overloaded experts
    // This allows the model to learn naturally imbalanced regime distributions
  }

  // Log for debugging (only occasionally to avoid spam)
  if (iter_ % 10 == 0) {
    std::string load_str = "";
    for (int k = 0; k < num_experts_; ++k) {
      load_str += std::to_string(actual_load[k]).substr(0, 5) + " ";
    }
    Log::Debug("MixtureGBDT: Expert loads = [%s], bias = [%.3f, %.3f, ...]",
               load_str.c_str(), expert_bias_[0],
               num_experts_ > 1 ? expert_bias_[1] : 0.0);
  }
}

void MixtureGBDT::MStepExperts() {
  const label_t* labels = train_data_->metadata().label();

  // Each expert should optimize its OWN prediction toward the label
  // Gradient for expert k: r_ik * d_L(y_i, f_k(x_i)) / d_f_k(x_i)
  // This ensures experts specialize on different parts of the data

  // Expert Dropout: randomly skip some experts to prevent collapse
  // This forces all experts to be useful by ensuring no single expert
  // can dominate. Dropped experts receive zero gradients for this iteration.
  const double dropout_rate = config_->mixture_expert_dropout_rate;
  std::vector<bool> expert_active(num_experts_, true);
  int num_active = num_experts_;

  if (dropout_rate > 0.0) {
    // Determine which experts to drop
    for (int k = 0; k < num_experts_; ++k) {
      if (dropout_dist_(dropout_rng_) < dropout_rate) {
        expert_active[k] = false;
        --num_active;
      }
    }

    // Ensure at least one expert is active
    // If all would be dropped, randomly select one to keep
    if (num_active == 0) {
      std::uniform_int_distribution<int> expert_dist(0, num_experts_ - 1);
      int keep_expert = expert_dist(dropout_rng_);
      expert_active[keep_expert] = true;
      num_active = 1;
    }

    // Log dropout info occasionally
    if (iter_ % 10 == 0) {
      std::string active_str = "";
      for (int k = 0; k < num_experts_; ++k) {
        active_str += (expert_active[k] ? "1" : "0");
        if (k < num_experts_ - 1) active_str += ",";
      }
      Log::Debug("MixtureGBDT: Expert dropout active (rate=%.2f), "
                 "active experts=[%s] (%d/%d)",
                 dropout_rate, active_str.c_str(), num_active, num_experts_);
    }
  }

  // Train each expert with responsibility-weighted gradients
  // computed from the expert's OWN prediction (not mixture yhat)
  for (int k = 0; k < num_experts_; ++k) {
    // Skip dropped experts (they receive zero gradients)
    if (!expert_active[k]) {
      // Still need to call TrainOneIter with zero gradients to maintain iteration count
      // But the tree will be a trivial no-split tree
      std::vector<score_t> zero_grad(num_data_, 0.0);
      std::vector<score_t> zero_hess(num_data_, kMixtureEpsilon);  // Small hessian to avoid numerical issues
      experts_[k]->TrainOneIter(zero_grad.data(), zero_hess.data());
      continue;
    }

    std::vector<score_t> grad_k(num_data_);
    std::vector<score_t> hess_k(num_data_);

    // Get expert k's predictions
    const double* expert_k_pred = expert_pred_.data() + k * num_data_;

    if (objective_function_ != nullptr) {
      // Use objective function to compute gradients for expert k's predictions
      std::vector<double> expert_k_pred_vec(expert_k_pred, expert_k_pred + num_data_);
      std::vector<score_t> temp_grad(num_data_);
      std::vector<score_t> temp_hess(num_data_);
      objective_function_->GetGradients(expert_k_pred_vec.data(), temp_grad.data(), temp_hess.data());

      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        grad_k[i] = static_cast<score_t>(r_ik * temp_grad[i]);
        hess_k[i] = static_cast<score_t>(r_ik * temp_hess[i]);
      }
    } else {
      // Default to MSE: d/df (y - f)^2 = 2*(f - y)
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        double diff = expert_k_pred[i] - labels[i];
        grad_k[i] = static_cast<score_t>(r_ik * 2.0 * diff);
        hess_k[i] = static_cast<score_t>(r_ik * 2.0);
      }
    }

    // Train one iteration with custom gradients
    experts_[k]->TrainOneIter(grad_k.data(), hess_k.data());
  }
}

void MixtureGBDT::MStepGate() {
  // Create pseudo-labels: z_i = argmax_k r_ik
  std::vector<label_t> pseudo_labels(num_data_);

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int best_k = 0;
    double best_r = responsibilities_[i * num_experts_];
    for (int k = 1; k < num_experts_; ++k) {
      double r = responsibilities_[i * num_experts_ + k];
      if (r > best_r) {
        best_r = r;
        best_k = k;
      }
    }
    pseudo_labels[i] = static_cast<label_t>(best_k);
  }

  // Update gate's labels
  // Note: This requires modifying the dataset's labels, which is complex.
  // For now, we'll use the gate's TrainOneIter with custom gradients.
  // TODO(shiyu1994): Implement proper label update for gate training

  // For multiclass, we compute softmax cross-entropy gradients
  std::vector<score_t> gate_grad(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<score_t> gate_hess(static_cast<size_t>(num_data_) * num_experts_);

  // Gate Entropy Regularization:
  // Encourages gate to produce more uniform (uncertain) predictions
  // This helps prevent premature expert collapse where gate assigns all samples to one expert
  //
  // Entropy H(g) = -Σ g_k log(g_k) is maximized when g_k = 1/K (uniform)
  // We add a regularization term that pushes probabilities toward uniform:
  // grad_reg = λ * (p_k - 1/K)
  // This makes the gate less confident early in training, allowing experts to differentiate
  const double entropy_lambda = config_->mixture_gate_entropy_lambda;
  const double uniform_prob = 1.0 / num_experts_;

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int label = static_cast<int>(pseudo_labels[i]);
    for (int k = 0; k < num_experts_; ++k) {
      size_t idx = i + k * num_data_;  // Gate uses class-major order
      double p = gate_proba_[i * num_experts_ + k];

      // Base gradient: softmax cross-entropy
      double base_grad;
      if (k == label) {
        base_grad = p - 1.0;
      } else {
        base_grad = p;
      }

      // Entropy regularization: push toward uniform distribution
      // grad_reg = λ * (p - 1/K)
      // When p > 1/K (over-confident), this adds positive gradient to reduce p
      // When p < 1/K (under-confident), this adds negative gradient to increase p
      double entropy_reg = entropy_lambda * (p - uniform_prob);

      gate_grad[idx] = static_cast<score_t>(base_grad + entropy_reg);

      // Hessian for softmax cross-entropy: p * (1 - p)
      // Note: We don't modify hessian for entropy regularization (would be λ, constant)
      // The constant hessian is implicitly absorbed into the learning rate
      gate_hess[idx] = static_cast<score_t>(std::max(p * (1.0 - p), kMixtureEpsilon));
    }
  }

  // Log entropy regularization effect (occasionally)
  if (entropy_lambda > 0.0 && iter_ % 10 == 0) {
    // Compute average entropy for monitoring
    double total_entropy = 0.0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      double sample_entropy = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        double p = gate_proba_[i * num_experts_ + k];
        if (p > kMixtureEpsilon) {
          sample_entropy -= p * std::log(p);
        }
      }
      total_entropy += sample_entropy;
    }
    double avg_entropy = total_entropy / num_data_;
    double max_entropy = std::log(static_cast<double>(num_experts_));
    double normalized_entropy = avg_entropy / max_entropy;
    Log::Debug("MixtureGBDT: Gate entropy regularization active (lambda=%.3f), "
               "avg normalized entropy=%.3f", entropy_lambda, normalized_entropy);
  }

  // Train gate for specified iterations
  for (int g = 0; g < config_->mixture_gate_iters_per_round; ++g) {
    gate_->TrainOneIter(gate_grad.data(), gate_hess.data());
  }
}

bool MixtureGBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  // MixtureGBDT ignores external gradients/hessians
  // (custom objective is handled internally via responsibility weighting)
  (void)gradients;
  (void)hessians;

  Log::Debug("MixtureGBDT::TrainOneIter - starting iteration %d", iter_);

  // Forward pass - compute expert predictions and gate probabilities
  Forward();

  // E-step: update responsibilities
  // Skip E-step for first few iterations to allow experts to differentiate
  // with the initial (non-uniform) responsibilities.
  // Without this, all experts start with prediction=0, causing EStep to
  // compute uniform responsibilities and all experts train identically.
  const int warmup_iters = config_->mixture_warmup_iters;
  if (iter_ >= warmup_iters) {
    EStep();

    // Apply time-series smoothing if enabled
    SmoothResponsibilities();

    // Update expert bias for loss-free load balancing
    UpdateExpertBias();
  }

  // M-step: update experts
  MStepExperts();

  // M-step: update gate
  // Gate should always be trained, even during warmup.
  // During warmup, responsibilities are fixed from initialization (quantile-based),
  // so gate learns to predict these fixed responsibilities.
  // This is crucial for hard alpha (1.0) to work properly.
  MStepGate();

  ++iter_;

  // Update validation predictions
  for (size_t v = 0; v < valid_datas_.size(); ++v) {
    ForwardValid(static_cast<int>(v));
  }

  Log::Debug("MixtureGBDT::TrainOneIter - completed iteration %d", iter_);

  // Check if we should continue
  // For now, always continue
  return false;
}

void MixtureGBDT::Train(int snapshot_freq, const std::string& model_output_path) {
  auto start_time = std::chrono::steady_clock::now();
  bool is_finished = false;

  for (int iter = 0; iter < config_->num_iterations && !is_finished; ++iter) {
    is_finished = TrainOneIter(nullptr, nullptr);

    // Check early stopping if validation data exists
    if (!is_finished && !valid_datas_.empty()) {
      is_finished = EvalAndCheckEarlyStopping();
    }

    auto end_time = std::chrono::steady_clock::now();
    Log::Info("MixtureGBDT: %f seconds elapsed, finished iteration %d",
              std::chrono::duration<double, std::milli>(end_time - start_time).count() * 1e-3,
              iter + 1);

    if (snapshot_freq > 0 && (iter + 1) % snapshot_freq == 0) {
      std::string snapshot_out = model_output_path + ".snapshot_iter_" + std::to_string(iter + 1);
      SaveModelToFile(0, -1, 0, snapshot_out.c_str());
    }
  }
}

int MixtureGBDT::GetCurrentIteration() const {
  return iter_;
}

void MixtureGBDT::RollbackOneIter() {
  if (iter_ > 0) {
    for (int k = 0; k < num_experts_; ++k) {
      experts_[k]->RollbackOneIter();
    }
    gate_->RollbackOneIter();
    --iter_;
  }
}

const double* MixtureGBDT::GetTrainingScore(int64_t* out_len) {
  *out_len = num_data_;
  return yhat_.data();
}

std::vector<double> MixtureGBDT::GetEvalAt(int data_idx) const {
  CHECK(data_idx >= 0 && data_idx <= static_cast<int>(valid_datas_.size()));
  std::vector<double> ret;

  if (data_idx == 0) {
    // Training data metrics
    for (const auto* metric : training_metrics_) {
      auto scores = metric->Eval(yhat_.data(), objective_function_);
      for (double score : scores) {
        ret.push_back(score);
      }
    }
  } else {
    // Validation data metrics
    int valid_idx = data_idx - 1;
    CHECK(valid_idx < static_cast<int>(valid_metrics_.size()));
    CHECK(valid_idx < static_cast<int>(yhat_valid_.size()));

    for (const auto* metric : valid_metrics_[valid_idx]) {
      auto scores = metric->Eval(yhat_valid_[valid_idx].data(), objective_function_);
      for (double score : scores) {
        ret.push_back(score);
      }
    }
  }
  return ret;
}

int64_t MixtureGBDT::GetNumPredictAt(int data_idx) const {
  if (data_idx == 0) {
    return num_data_;
  }
  int valid_idx = data_idx - 1;
  CHECK(valid_idx >= 0 && valid_idx < static_cast<int>(valid_datas_.size()));
  return valid_datas_[valid_idx]->num_data();
}

void MixtureGBDT::GetPredictAt(int data_idx, double* result, int64_t* out_len) {
  if (data_idx == 0) {
    std::copy(yhat_.begin(), yhat_.end(), result);
    *out_len = num_data_;
  } else {
    int valid_idx = data_idx - 1;
    CHECK(valid_idx >= 0 && valid_idx < static_cast<int>(yhat_valid_.size()));
    const auto& yhat = yhat_valid_[valid_idx];
    std::copy(yhat.begin(), yhat.end(), result);
    *out_len = static_cast<int64_t>(yhat.size());
  }
}

std::string MixtureGBDT::OutputMetric(int iter) {
  bool need_output = (iter % config_->metric_freq) == 0;
  std::string ret = "";
  std::stringstream msg_buf;
  std::vector<std::pair<size_t, size_t>> meet_early_stopping_pairs;

  // Print training metrics
  if (need_output) {
    for (const auto* metric : training_metrics_) {
      auto name = metric->GetName();
      auto scores = metric->Eval(yhat_.data(), objective_function_);
      for (size_t k = 0; k < name.size(); ++k) {
        Log::Info("Iteration:%d, training %s : %f", iter, name[k].c_str(), scores[k]);
        if (early_stopping_round_ > 0) {
          msg_buf << "Iteration:" << iter << ", training " << name[k] << " : " << scores[k] << '\n';
        }
      }
    }
  }

  // Print and check validation metrics
  if (need_output || early_stopping_round_ > 0) {
    for (size_t i = 0; i < valid_metrics_.size(); ++i) {
      for (size_t j = 0; j < valid_metrics_[i].size(); ++j) {
        auto scores = valid_metrics_[i][j]->Eval(yhat_valid_[i].data(), objective_function_);
        auto name = valid_metrics_[i][j]->GetName();

        for (size_t k = 0; k < name.size(); ++k) {
          std::stringstream tmp_buf;
          tmp_buf << "Iteration:" << iter << ", valid_" << (i + 1) << " " << name[k] << " : " << scores[k];
          if (need_output) {
            Log::Info(tmp_buf.str().c_str());
          }
          if (early_stopping_round_ > 0) {
            msg_buf << tmp_buf.str() << '\n';
          }
        }

        // Check early stopping
        if (es_first_metric_only_ && j > 0) {
          continue;
        }

        if (ret.empty() && early_stopping_round_ > 0 &&
            i < best_score_.size() && j < best_score_[i].size()) {
          double cur_score = valid_metrics_[i][j]->factor_to_bigger_better() * scores.back();
          if (cur_score - best_score_[i][j] > early_stopping_min_delta_) {
            // Found a better score
            best_score_[i][j] = cur_score;
            best_iter_[i][j] = iter;
            meet_early_stopping_pairs.emplace_back(i, j);
          } else if (iter - best_iter_[i][j] >= early_stopping_round_) {
            // No improvement for early_stopping_round_ iterations
            ret = best_msg_[i][j];
          }
        }
      }
    }
  }

  // Update best messages
  for (auto& pair : meet_early_stopping_pairs) {
    best_msg_[pair.first][pair.second] = msg_buf.str();
  }
  return ret;
}

bool MixtureGBDT::EvalAndCheckEarlyStopping() {
  bool is_met_early_stopping = false;
  std::string best_msg_str = OutputMetric(iter_);

  is_met_early_stopping = !best_msg_str.empty();
  if (is_met_early_stopping) {
    Log::Info("Early stopping at iteration %d, the best iteration round is %d",
              iter_, iter_ - early_stopping_round_);
    Log::Info("Output of best iteration round:\n%s", best_msg_str.c_str());

    // Rollback models to best iteration
    for (int i = 0; i < early_stopping_round_; ++i) {
      RollbackOneIter();
    }
  }
  return is_met_early_stopping;
}

int MixtureGBDT::NumPredictOneRow(int start_iteration, int num_iteration,
                                   bool is_pred_leaf, bool is_pred_contrib) const {
  (void)start_iteration;
  (void)num_iteration;
  (void)is_pred_leaf;
  (void)is_pred_contrib;
  // Return 1 for the combined prediction
  return 1;
}

void MixtureGBDT::Predict(const double* features, double* output,
                          const PredictionEarlyStopInstance* earlyStop) const {
  // Create a no-op early stop instance if none provided
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = earlyStop ? earlyStop : &no_early_stop;

  // Get expert predictions
  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &expert_preds[k], early_stop_ptr);
  }

  // Get gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), early_stop_ptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Compute weighted sum
  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRaw(const double* features, double* output,
                             const PredictionEarlyStopInstance* earlyStop) const {
  Predict(features, output, earlyStop);
}

void MixtureGBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output,
                               const PredictionEarlyStopInstance* early_stop) const {
  // Create a no-op early stop instance if none provided
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = early_stop ? early_stop : &no_early_stop;

  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->PredictByMap(features, &expert_preds[k], early_stop_ptr);
  }

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRawByMap(features, gate_raw.data(), early_stop_ptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                                  const PredictionEarlyStopInstance* early_stop) const {
  PredictByMap(features, output, early_stop);
}

void MixtureGBDT::PredictRegime(const double* features, int* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Find argmax
  int best_k = 0;
  double best_p = gate_prob[0];
  for (int k = 1; k < num_experts_; ++k) {
    if (gate_prob[k] > best_p) {
      best_p = gate_prob[k];
      best_k = k;
    }
  }
  *output = best_k;
}

void MixtureGBDT::PredictRegimeProba(const double* features, double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);
  Softmax(gate_raw.data(), num_experts_, output);
}

void MixtureGBDT::PredictExpertPred(const double* features, double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &output[k], &no_early_stop);
  }
}

void MixtureGBDT::PredictWithPrevProba(const double* features, const double* prev_proba,
                                        double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());
  const PredictionEarlyStopInstance* early_stop_ptr = &no_early_stop;

  // Get expert predictions
  std::vector<double> expert_preds(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->Predict(features, &expert_preds[k], early_stop_ptr);
  }

  // Get current gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), early_stop_ptr);

  std::vector<double> gate_prob(num_experts_);
  Softmax(gate_raw.data(), num_experts_, gate_prob.data());

  // Blend with prev_proba if provided and in Markov mode
  if (use_markov_ && prev_proba != nullptr) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        gate_prob[k] = (1.0 - lambda) * gate_prob[k] + lambda * prev_proba[k];
        sum += gate_prob[k];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        gate_prob[k] /= sum;
      }
    }
  }

  // Compute weighted sum
  double sum = 0.0;
  for (int k = 0; k < num_experts_; ++k) {
    sum += gate_prob[k] * expert_preds[k];
  }
  *output = sum;
}

void MixtureGBDT::PredictRegimeProbaWithPrevProba(const double* features, const double* prev_proba,
                                                   double* output) const {
  // Create a no-op early stop instance
  PredictionEarlyStopInstance no_early_stop = CreatePredictionEarlyStopInstance(
      "none", PredictionEarlyStopConfig());

  // Get current gate probabilities
  std::vector<double> gate_raw(num_experts_);
  gate_->PredictRaw(features, gate_raw.data(), &no_early_stop);
  Softmax(gate_raw.data(), num_experts_, output);

  // Blend with prev_proba if provided and in Markov mode
  if (use_markov_ && prev_proba != nullptr) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0) {
      double sum = 0.0;
      for (int k = 0; k < num_experts_; ++k) {
        output[k] = (1.0 - lambda) * output[k] + lambda * prev_proba[k];
        sum += output[k];
      }
      // Renormalize
      for (int k = 0; k < num_experts_; ++k) {
        output[k] /= sum;
      }
    }
  }
}

void MixtureGBDT::PredictLeafIndex(const double* features, double* output) const {
  (void)features;
  (void)output;
  Log::Fatal("MixtureGBDT::PredictLeafIndex is not implemented");
}

void MixtureGBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features,
                                        double* output) const {
  (void)features;
  (void)output;
  Log::Fatal("MixtureGBDT::PredictLeafIndexByMap is not implemented");
}

void MixtureGBDT::PredictContrib(const double* features, double* output) const {
  (void)features;
  (void)output;
  Log::Fatal("MixtureGBDT::PredictContrib (SHAP) is not implemented");
}

void MixtureGBDT::PredictContribByMap(const std::unordered_map<int, double>& features,
                                       std::vector<std::unordered_map<int, double>>* output) const {
  (void)features;
  (void)output;
  Log::Fatal("MixtureGBDT::PredictContribByMap is not implemented");
}

void MixtureGBDT::MergeFrom(const Boosting* other) {
  (void)other;
  Log::Fatal("MixtureGBDT::MergeFrom is not implemented");
}

void MixtureGBDT::ShuffleModels(int start_iter, int end_iter) {
  (void)start_iter;
  (void)end_iter;
  Log::Warning("MixtureGBDT::ShuffleModels is not supported");
}

void MixtureGBDT::ResetTrainingData(const Dataset* train_data,
                                    const ObjectiveFunction* objective_function,
                                    const std::vector<const Metric*>& training_metrics) {
  train_data_ = train_data;
  objective_function_ = objective_function;
  training_metrics_ = training_metrics;
  num_data_ = train_data_->num_data();

  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->ResetTrainingData(train_data_, objective_function_, training_metrics_);
  }
  gate_->ResetTrainingData(train_data_, nullptr, {});

  // Reallocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  gate_proba_.resize(nk);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  // Reset expert bias for loss-free load balancing
  std::fill(expert_bias_.begin(), expert_bias_.end(), 0.0);

  InitResponsibilities();
}

void MixtureGBDT::ResetConfig(const Config* config) {
  config_ = std::unique_ptr<Config>(new Config(*config));
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->ResetConfig(config);
  }
  gate_->ResetConfig(gate_config_.get());
}

void MixtureGBDT::AddValidDataset(const Dataset* valid_data,
                                  const std::vector<const Metric*>& valid_metrics) {
  // Validate dataset alignment with training data
  if (!train_data_->CheckAlign(*valid_data)) {
    Log::Fatal("Cannot add validation data, since it has different bin mappers with training data");
  }

  // Store validation data and metrics
  valid_datas_.push_back(valid_data);
  valid_metrics_.push_back(valid_metrics);

  // Get validation data size
  data_size_t num_valid = valid_data->num_data();
  size_t nk_valid = static_cast<size_t>(num_valid) * num_experts_;

  // Allocate validation buffers for this dataset
  expert_pred_valid_.emplace_back(nk_valid, 0.0);
  gate_proba_valid_.emplace_back(nk_valid, 0.0);
  yhat_valid_.emplace_back(num_valid, 0.0);

  // Initialize Markov buffers if needed
  if (use_markov_) {
    const double uniform_prob = 1.0 / num_experts_;
    prev_gate_proba_valid_.emplace_back(nk_valid, uniform_prob);
  }

  // Add validation data to each expert and gate (they will manage their own ScoreUpdaters)
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->AddValidDataset(valid_data, {});  // No metrics at expert level
  }
  gate_->AddValidDataset(valid_data, {});  // No metrics at gate level

  // Compute initial validation predictions if we have trained iterations
  if (iter_ > 0) {
    ForwardValid(static_cast<int>(valid_datas_.size()) - 1);
  }

  // Initialize early stopping tracking if enabled
  if (early_stopping_round_ > 0) {
    size_t num_metrics = valid_metrics.size();
    if (es_first_metric_only_) {
      num_metrics = std::min(num_metrics, static_cast<size_t>(1));
    }
    best_iter_.emplace_back(num_metrics, 0);
    best_score_.emplace_back(num_metrics, -std::numeric_limits<double>::infinity());
    best_msg_.emplace_back(num_metrics);
  }

  Log::Info("MixtureGBDT: Added validation dataset %d with %d samples",
            static_cast<int>(valid_datas_.size()), num_valid);
}

void MixtureGBDT::RefitTree(const int* tree_leaf_prediction, const size_t nrow, const size_t ncol) {
  (void)tree_leaf_prediction;
  (void)nrow;
  (void)ncol;
  Log::Fatal("MixtureGBDT::RefitTree is not implemented");
}

std::string MixtureGBDT::DumpModel(int start_iteration, int num_iteration,
                                   int feature_importance_type) const {
  (void)start_iteration;
  (void)num_iteration;
  (void)feature_importance_type;
  Log::Fatal("MixtureGBDT::DumpModel is not implemented");
  return "{}";
}

std::string MixtureGBDT::ModelToIfElse(int num_iteration) const {
  (void)num_iteration;
  Log::Fatal("MixtureGBDT::ModelToIfElse is not implemented");
  return "";
}

bool MixtureGBDT::SaveModelToIfElse(int num_iteration, const char* filename) const {
  (void)num_iteration;
  (void)filename;
  Log::Fatal("MixtureGBDT::SaveModelToIfElse is not implemented");
  return false;
}

bool MixtureGBDT::SaveModelToFile(int start_iteration, int num_iterations,
                                   int feature_importance_type, const char* filename) const {
  std::string model_str = SaveModelToString(start_iteration, num_iterations, feature_importance_type);
  if (model_str.empty()) {
    return false;
  }
  std::ofstream file(filename);
  if (!file.is_open()) {
    return false;
  }
  file << model_str;
  return true;
}

std::string MixtureGBDT::SaveModelToString(int start_iteration, int num_iterations,
                                            int feature_importance_type) const {
  std::stringstream ss;

  // Mixture header
  ss << "mixture\n";
  ss << "mixture_enable=1\n";
  ss << "mixture_num_experts=" << num_experts_ << "\n";
  ss << "mixture_e_step_alpha=" << config_->mixture_e_step_alpha << "\n";
  ss << "mixture_e_step_loss=" << e_step_loss_type_ << "\n";
  ss << "mixture_e_step_mode=" << config_->mixture_e_step_mode << "\n";
  ss << "mixture_r_smoothing=" << config_->mixture_r_smoothing << "\n";
  ss << "mixture_smoothing_lambda=" << config_->mixture_smoothing_lambda << "\n";
  ss << "\n";

  // Gate model
  ss << "[gate_model]\n";
  ss << gate_->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
  ss << "\n";

  // Expert models
  for (int k = 0; k < num_experts_; ++k) {
    ss << "[expert_model_" << k << "]\n";
    ss << experts_[k]->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
    ss << "\n";
  }

  return ss.str();
}

bool MixtureGBDT::LoadModelFromString(const char* buffer, size_t len) {
  std::string model_str(buffer, len);
  std::istringstream ss(model_str);
  std::string line;

  // Read header (trim to handle Windows CRLF line endings)
  if (!std::getline(ss, line) || Common::Trim(line) != "mixture") {
    Log::Fatal("Invalid mixture model format: expected 'mixture' header");
    return false;
  }

  // Parse mixture parameters
  std::unordered_map<std::string, std::string> params;
  while (std::getline(ss, line)) {
    line = Common::Trim(line);
    if (line.empty() || line[0] == '[') {
      break;
    }
    size_t eq_pos = line.find('=');
    if (eq_pos != std::string::npos) {
      std::string key = Common::Trim(line.substr(0, eq_pos));
      std::string value = Common::Trim(line.substr(eq_pos + 1));
      params[key] = value;
    }
  }

  // Extract parameters
  if (params.count("mixture_num_experts")) {
    num_experts_ = std::stoi(params["mixture_num_experts"]);
  }
  if (params.count("mixture_e_step_loss")) {
    e_step_loss_type_ = params["mixture_e_step_loss"];
  }

  // Store loaded parameters for GetLoadedParam (must be valid JSON)
  std::stringstream param_ss;
  param_ss << "{";
  bool first = true;
  for (const auto& kv : params) {
    if (!first) {
      param_ss << ", ";
    }
    first = false;
    param_ss << "\"" << kv.first << "\": ";
    // Try to detect numeric values
    bool is_numeric = !kv.second.empty() && (std::isdigit(kv.second[0]) || kv.second[0] == '-' || kv.second[0] == '.');
    if (is_numeric) {
      param_ss << kv.second;
    } else {
      param_ss << "\"" << kv.second << "\"";
    }
  }
  param_ss << "}";
  loaded_parameter_ = param_ss.str();

  // Find sections
  std::string gate_model_str;
  std::vector<std::string> expert_model_strs(num_experts_);

  // We need to re-parse from the section markers
  std::string remaining = model_str.substr(ss.tellg() > 0 ? static_cast<size_t>(ss.tellg()) - line.size() - 1 : 0);

  // Find [gate_model]
  size_t gate_start = remaining.find("[gate_model]");
  if (gate_start == std::string::npos) {
    Log::Fatal("Invalid mixture model format: [gate_model] section not found");
    return false;
  }
  gate_start += std::string("[gate_model]").length();
  // Skip newline (handles both CRLF and LF)
  if (gate_start < remaining.length() && remaining[gate_start] == '\r') {
    ++gate_start;
  }
  if (gate_start < remaining.length() && remaining[gate_start] == '\n') {
    ++gate_start;
  }

  // Find first expert
  size_t expert0_start = remaining.find("[expert_model_0]");
  if (expert0_start == std::string::npos) {
    Log::Fatal("Invalid mixture model format: [expert_model_0] section not found");
    return false;
  }

  gate_model_str = remaining.substr(gate_start, expert0_start - gate_start);

  // Parse expert models
  for (int k = 0; k < num_experts_; ++k) {
    std::string section_name = "[expert_model_" + std::to_string(k) + "]";
    size_t section_start = remaining.find(section_name);
    if (section_start == std::string::npos) {
      Log::Fatal("Invalid mixture model format: %s section not found", section_name.c_str());
      return false;
    }
    section_start += section_name.length();
    // Skip newline (handles both CRLF and LF)
    if (section_start < remaining.length() && remaining[section_start] == '\r') {
      ++section_start;
    }
    if (section_start < remaining.length() && remaining[section_start] == '\n') {
      ++section_start;
    }

    // Find next section or end
    size_t section_end;
    if (k < num_experts_ - 1) {
      std::string next_section = "[expert_model_" + std::to_string(k + 1) + "]";
      section_end = remaining.find(next_section);
    } else {
      section_end = remaining.length();
    }

    expert_model_strs[k] = remaining.substr(section_start, section_end - section_start);
  }

  // Create config for loading (minimal config)
  config_ = std::unique_ptr<Config>(new Config());
  config_->mixture_num_experts = num_experts_;

  // Initialize experts
  experts_.clear();
  experts_.reserve(num_experts_);
  for (int k = 0; k < num_experts_; ++k) {
    experts_.emplace_back(new GBDT());
    if (!experts_[k]->LoadModelFromString(expert_model_strs[k].c_str(), expert_model_strs[k].size())) {
      Log::Fatal("Failed to load expert model %d", k);
      return false;
    }
  }

  // Initialize gate
  gate_.reset(new GBDT());
  if (!gate_->LoadModelFromString(gate_model_str.c_str(), gate_model_str.size())) {
    Log::Fatal("Failed to load gate model");
    return false;
  }

  // Set feature info from first expert
  if (!experts_.empty()) {
    max_feature_idx_ = experts_[0]->MaxFeatureIdx();
    feature_names_ = experts_[0]->FeatureNames();
    label_idx_ = experts_[0]->LabelIdx();
    // Set iteration count from first expert
    iter_ = experts_[0]->GetCurrentIteration();
  }

  Log::Info("MixtureGBDT: Loaded model with %d experts, %d iterations", num_experts_, iter_);
  return true;
}

std::vector<double> MixtureGBDT::FeatureImportance(int num_iteration, int importance_type) const {
  // Sum importance across all experts
  std::vector<double> result(max_feature_idx_ + 1, 0.0);

  for (int k = 0; k < num_experts_; ++k) {
    auto expert_imp = experts_[k]->FeatureImportance(num_iteration, importance_type);
    for (size_t i = 0; i < expert_imp.size() && i < result.size(); ++i) {
      result[i] += expert_imp[i];
    }
  }

  return result;
}

double MixtureGBDT::GetUpperBoundValue() const {
  double max_val = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < num_experts_; ++k) {
    max_val = std::max(max_val, experts_[k]->GetUpperBoundValue());
  }
  return max_val;
}

double MixtureGBDT::GetLowerBoundValue() const {
  double min_val = std::numeric_limits<double>::infinity();
  for (int k = 0; k < num_experts_; ++k) {
    min_val = std::min(min_val, experts_[k]->GetLowerBoundValue());
  }
  return min_val;
}

int MixtureGBDT::MaxFeatureIdx() const {
  return max_feature_idx_;
}

std::vector<std::string> MixtureGBDT::FeatureNames() const {
  return feature_names_;
}

int MixtureGBDT::LabelIdx() const {
  return label_idx_;
}

int MixtureGBDT::NumberOfTotalModel() const {
  int total = 0;
  for (int k = 0; k < num_experts_; ++k) {
    total += experts_[k]->NumberOfTotalModel();
  }
  total += gate_->NumberOfTotalModel();
  return total;
}

int MixtureGBDT::NumModelPerIteration() const {
  // One tree per expert per iteration, plus gate trees
  return num_experts_ + gate_->NumModelPerIteration();
}

int MixtureGBDT::NumberOfClasses() const {
  return 1;  // Regression output
}

bool MixtureGBDT::NeedAccuratePrediction() const {
  return true;
}

void MixtureGBDT::InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) {
  for (int k = 0; k < num_experts_; ++k) {
    experts_[k]->InitPredict(start_iteration, num_iteration, is_pred_contrib);
  }
  gate_->InitPredict(start_iteration, num_iteration, is_pred_contrib);
}

double MixtureGBDT::GetLeafValue(int tree_idx, int leaf_idx) const {
  (void)tree_idx;
  (void)leaf_idx;
  Log::Fatal("MixtureGBDT::GetLeafValue is not implemented for mixture models");
  return 0.0;  // Unreachable, but needed for compiler
}

void MixtureGBDT::SetLeafValue(int tree_idx, int leaf_idx, double val) {
  (void)tree_idx;
  (void)leaf_idx;
  (void)val;
  Log::Fatal("MixtureGBDT::SetLeafValue is not implemented for mixture models");
}

std::string MixtureGBDT::GetLoadedParam() const {
  Log::Warning("MixtureGBDT::GetLoadedParam called, loaded_parameter_=%s", loaded_parameter_.c_str());
  return loaded_parameter_;
}

std::string MixtureGBDT::ParserConfigStr() const {
  if (!experts_.empty()) {
    return experts_[0]->ParserConfigStr();
  }
  return "";
}

}  // namespace LightGBM
