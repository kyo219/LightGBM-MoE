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
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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
      use_progressive_(false),
      seed_iterations_(0),
      seed_phase_complete_(false),
      gate_temperature_(1.0),
      early_stopping_round_(0),
      early_stopping_min_delta_(0.0),
      es_first_metric_only_(false) {
}

MixtureGBDT::~MixtureGBDT() {
  for (BinIterator* p : leaf_reuse_iters_) {
    delete p;
  }
}

void MixtureGBDT::Init(const Config* config, const Dataset* train_data,
                       const ObjectiveFunction* objective_function,
                       const std::vector<const Metric*>& training_metrics) {
  CHECK_NOTNULL(train_data);
  train_data_ = train_data;
  objective_function_ = objective_function;
  training_metrics_ = training_metrics;
  num_data_ = train_data_->num_data();

  // leaf_reuse gate uses bin data directly (no raw features needed)
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
  //
  // Disable per-expert bagging. With hard M-step sparse activation (#10) we
  // already restrict each expert to its assigned samples via SetBaggingData
  // from MStepExperts; if the user also sets bagging_fraction<1.0 / bagging_freq>0
  // then GBDT::TrainOneIter's internal Bagging() overwrites our restriction
  // with a random bag of the *full* dataset. Combined with the sparse-gradient
  // vector we hand the expert (zero grad for unassigned samples), this produces
  // degenerate per-bin histograms where the "best" split has one side empty —
  // the LightGBM split finder then trips CHECK_GT(left_count/right_count, 0)
  // (issue #16; doesn't reproduce on standard GBDT because every sample carries
  // a real gradient there). Sparse activation is MoE's equivalent of bagging.
  expert_config_->bagging_fraction = 1.0;
  expert_config_->bagging_freq = 0;
  expert_config_->pos_bagging_fraction = 1.0;
  expert_config_->neg_bagging_fraction = 1.0;

  // Force leaf-output renewal whenever experts run with quantized gradients.
  // Sparse activation (hard M-step) gives unassigned samples hess≈1e-12 — the
  // quantized leaf-value path then derives leaf outputs from int histogram
  // sums × grad/hess scales computed over the full vector, which produces
  // systematically biased outputs (RMSE blows up by ~3-20x in the bench).
  // Renewing leaf outputs from the original float gradients restores accuracy
  // while keeping the quantized split-finding speedup.
  if (expert_config_->use_quantized_grad) {
    expert_config_->quant_train_renew_leaf = true;
    Log::Info("MixtureGBDT: use_quantized_grad=true, forcing quant_train_renew_leaf=true "
              "on experts (mitigates RMSE regression with hard M-step sparse activation)");
  }

  // Create gate config (multiclass classification)
  gate_config_ = std::unique_ptr<Config>(new Config(*config));
  gate_config_->objective = "multiclass";
  gate_config_->num_class = num_experts_;
  gate_config_->max_depth = config_->mixture_gate_max_depth;
  gate_config_->num_leaves = config_->mixture_gate_num_leaves;
  gate_config_->learning_rate = config_->mixture_gate_learning_rate;
  gate_config_->lambda_l2 = config_->mixture_gate_lambda_l2;
  if (gate_config_->use_quantized_grad) {
    gate_config_->quant_train_renew_leaf = true;
  }

  // Progressive training mode (EvoMoE)
  use_progressive_ = (config_->mixture_progressive_mode == "evomoe");
  seed_iterations_ = config_->mixture_seed_iterations;
  seed_phase_complete_ = false;

  // Gate temperature annealing
  gate_temperature_ = config_->mixture_gate_temperature_init;

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
  const bool use_per_expert_extra_trees = !config_->mixture_expert_extra_trees.empty();

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
  if (use_per_expert_extra_trees &&
      static_cast<int>(config_->mixture_expert_extra_trees.size()) != num_experts_) {
    Log::Fatal("mixture_expert_extra_trees must have exactly %d values (one per expert), got %d",
               num_experts_, static_cast<int>(config_->mixture_expert_extra_trees.size()));
  }

  if (use_progressive_) {
    // EvoMoE progressive mode: create only a single seed expert
    Log::Info("MixtureGBDT: Progressive mode (EvoMoE) enabled, "
              "seed_iterations=%d, perturbation=%.2f",
              seed_iterations_, config_->mixture_spawn_perturbation);

    seed_expert_.reset(new GBDT());
    seed_expert_->Init(expert_config_.get(), train_data_, nullptr, {});

    // Don't create K experts yet - they will be spawned after seed phase
    // But we still need expert_configs_ for later use in SpawnExpertsFromSeed
    expert_configs_.clear();
    expert_configs_.reserve(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      expert_configs_.emplace_back(new Config(*expert_config_));
      expert_configs_[k]->seed = config_->seed + k + 1;

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
      if (use_per_expert_extra_trees) {
        expert_configs_[k]->extra_trees = (config_->mixture_expert_extra_trees[k] != 0);
      }
    }
  } else {
    // Standard mode: create K experts from scratch
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
      if (use_per_expert_extra_trees) {
        expert_configs_[k]->extra_trees = (config_->mixture_expert_extra_trees[k] != 0);
      }

      experts_.emplace_back(new GBDT());
      Log::Debug("MixtureGBDT::Init - initializing expert %d with seed %d, max_depth=%d, num_leaves=%d, min_data=%d, min_gain=%.4f, extra_trees=%d",
                 k, expert_configs_[k]->seed, expert_configs_[k]->max_depth,
                 expert_configs_[k]->num_leaves, expert_configs_[k]->min_data_in_leaf,
                 expert_configs_[k]->min_gain_to_split, expert_configs_[k]->extra_trees ? 1 : 0);
      experts_[k]->Init(expert_configs_[k].get(), train_data_, nullptr, {});
      Log::Debug("MixtureGBDT::Init - expert %d initialized", k);
    }
  }

  // Check smoothing modes
  use_markov_ = (config_->mixture_r_smoothing == "markov");
  use_momentum_ = (config_->mixture_r_smoothing == "momentum");

  // Initialize gate
  Log::Debug("MixtureGBDT::Init - creating gate");
  gate_.reset(new GBDT());
  Log::Debug("MixtureGBDT::Init - initializing gate");

  if (use_markov_) {
    Log::Info("MixtureGBDT: Markov mode enabled (lambda=%.2f). "
              "INFERENCE CONTRACT: standard Predict()/predict() returns "
              "un-smoothed routing — to match the routing used at training "
              "and validation, call Booster.predict_markov() / "
              "predict_regime_proba_markov() (Python) which sweep the "
              "Markov prior across the rows of the input. Without that, "
              "test-set metrics will silently disagree with val metrics "
              "selected during tuning.",
              config_->mixture_smoothing_lambda);
  } else if (use_momentum_) {
    Log::Info("MixtureGBDT: Momentum mode enabled (lambda=%.2f)",
              config_->mixture_smoothing_lambda);
  }

  // Warn when the legacy fixed-alpha E-step is enabled. With per-expert
  // variance estimation off, `score = log(π_k) − alpha·loss` makes alpha a
  // hyperparameter whose right value scales with the y-magnitude (Var(y)
  // for L2, |y| for L1), so the same alpha on (say) sp500 returns and a
  // synthetic-magnitude regression target produces wildly different
  // routing temperatures. Default behavior since PR #24 is variance
  // estimation on; users who explicitly turn it off should know the
  // alpha-tuning contract changes.
  if (!config_->mixture_estimate_variance) {
    Log::Warning(
        "MixtureGBDT: mixture_estimate_variance=false — falling back to the "
        "legacy `log(gate) − alpha * loss` E-step heuristic with fixed "
        "alpha=%g. alpha is a temperature on |residual|^p that depends on "
        "the y scale (alpha * loss = O(y_magnitude^p)), so the same alpha "
        "behaves very differently across datasets. Either keep the default "
        "(true) or tune mixture_e_step_alpha relative to Var(y) for L2 / "
        "E[|y − ymean|] for L1. ELBO logging is also disabled in this mode.",
        config_->mixture_e_step_alpha);
  }

  // Time-order guard for any mode whose responsibility / gate-proba update
  // shifts by row index. EMA / momentum / Markov smoothing all assume row i
  // is the temporal successor of row i-1. If the dataset is shuffled (which
  // is LightGBM's default for non-time-series problems and for any random
  // CV fold), these shifts blend unrelated samples and silently corrupt
  // routing. There is no reliable way to detect ordering from the in-memory
  // dataset, so we surface a loud warning instead.
  const bool order_dependent_smoothing =
      use_markov_ || use_momentum_ ||
      config_->mixture_r_smoothing == "ema";
  if (order_dependent_smoothing && config_->mixture_smoothing_lambda > 0.0) {
    Log::Warning(
        "MixtureGBDT: r_smoothing='%s' (lambda=%.2f) shifts responsibilities "
        "by row index — only valid if rows are in true temporal order. "
        "Random shuffling (default for many CV setups) will silently mix "
        "unrelated samples. Disable smoothing (lambda=0) for shuffled data.",
        config_->mixture_r_smoothing.c_str(),
        config_->mixture_smoothing_lambda);
  }
  gate_->Init(gate_config_.get(), train_data_, nullptr, {});
  Log::Debug("MixtureGBDT::Init - gate initialized");

  // Allocate buffers
  size_t nk = static_cast<size_t>(num_data_) * num_experts_;
  responsibilities_.resize(nk);
  expert_pred_.resize(nk);
  expert_pred_sm_.resize(nk);
  gate_proba_.resize(nk);
  // Bias-free routing prior used by E-step / ELBO / affinity. Initialized to
  // uniform 1/K so the very first EStep (if any happens before Forward) sees
  // a well-defined prior rather than zeros that would produce -inf log priors.
  gate_proba_no_bias_.assign(nk, 1.0 / num_experts_);
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

  // Initialize per-expert noise scale to the marginal residual variance.
  // Using the empirical variance of y as a proxy for the worst-case scale
  // ensures the first E-step does not divide by an absurdly small σ_k² for
  // experts that haven't yet predicted anything (all f_k start at 0).
  expert_variance_.resize(num_experts_, 1.0);
  if (config_->mixture_estimate_variance) {
    const label_t* init_labels = train_data_->metadata().label();
    if (init_labels != nullptr && num_data_ > 0) {
      double mean = 0.0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        mean += static_cast<double>(init_labels[i]);
      }
      mean /= num_data_;
      double var_or_scale = 0.0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double r = static_cast<double>(init_labels[i]) - mean;
        if (e_step_loss_type_ == "l1") {
          var_or_scale += std::fabs(r);  // Laplace b
        } else {
          var_or_scale += r * r;          // Gaussian σ²
        }
      }
      var_or_scale /= num_data_;
      var_or_scale = std::max(var_or_scale, kMixtureEpsilon);
      for (int k = 0; k < num_experts_; ++k) {
        expert_variance_[k] = var_or_scale;
      }
      Log::Info("MixtureGBDT: initial per-expert noise scale = %.4g (%s)",
                var_or_scale,
                e_step_loss_type_ == "l1" ? "Laplace b" : "Gaussian σ²");
    }
  }

  // Initialize expert load for auxiliary load balancing
  // Add small random perturbation to break initial symmetry
  expert_load_.resize(num_experts_);
  {
    std::mt19937 load_rng(config_->seed + 99999);
    std::uniform_real_distribution<double> noise_dist(-0.1, 0.1);
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      expert_load_[k] = 1.0 / num_experts_ + noise_dist(load_rng);
      expert_load_[k] = std::max(expert_load_[k], 0.01);  // Ensure positive
      sum += expert_load_[k];
    }
    // Normalize to sum to 1
    for (int k = 0; k < num_experts_; ++k) {
      expert_load_[k] /= sum;
    }
  }

  // Initialize expert dropout RNG
  dropout_rng_.seed(config_->seed + 12345);  // Different seed from other RNGs
  dropout_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);

  // Initialize adaptive per-expert learning rate tracking
  if (config_->mixture_adaptive_lr) {
    Log::Warning(
        "MixtureGBDT: mixture_adaptive_lr=true scales each expert's learning "
        "rate by its own loss trend, which means the K experts no longer "
        "share a joint EM objective. The marginal log-likelihood is no longer "
        "guaranteed to improve monotonically and ELBO diagnostics may be "
        "misleading. Disable for principled EM unless you have a specific "
        "reason to use it.");
    const int window = config_->mixture_adaptive_lr_window;
    expert_loss_history_.resize(num_experts_, std::vector<double>(window, 0.0));
    expert_lr_scale_.resize(num_experts_, 1.0);
    loss_history_pos_ = 0;
  }

  // Expert Choice Routing initialization
  use_expert_choice_ = (config_->mixture_routing_mode == "expert_choice");

  if (use_expert_choice_) {
    double capacity_ratio = static_cast<double>(num_data_) / num_experts_;
    expert_capacity_ = static_cast<int>(
        capacity_ratio * config_->mixture_expert_capacity_factor);
    expert_capacity_ = std::max(1, expert_capacity_);

    affinity_scores_.resize(nk);
    expert_selection_mask_.resize(nk);

    Log::Info("MixtureGBDT: Expert Choice Routing enabled "
              "(capacity=%d, factor=%.2f)",
              expert_capacity_, config_->mixture_expert_capacity_factor);
  }

  // Initialize early stopping parameters
  early_stopping_round_ = config_->early_stopping_round;
  early_stopping_min_delta_ = config_->early_stopping_min_delta;
  es_first_metric_only_ = config_->first_metric_only;

  // Initialize responsibilities (skip during progressive seed phase)
  if (!use_progressive_) {
    InitResponsibilities();
  }

  // Log temperature annealing config
  if (config_->mixture_gate_temperature_init != config_->mixture_gate_temperature_final) {
    Log::Info("MixtureGBDT: Gate temperature annealing enabled (init=%.2f, final=%.2f)",
              config_->mixture_gate_temperature_init, config_->mixture_gate_temperature_final);
  }

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
  } else if (config_->mixture_init == "balanced_kmeans") {
    // Balanced K-Means on (features, label). Each expert gets exactly N/K
    // samples. The label is concatenated as an extra dimension, which biases
    // clusters toward y-magnitude — useful when y is the strongest signal,
    // but does NOT discover regimes in X-space alone.
    Log::Info("MixtureGBDT: Using Balanced K-Means init on features + label");
    InitResponsibilitiesBalancedKMeans(labels, /*include_label=*/true);

  } else if (config_->mixture_init == "kmeans_features") {
    // Balanced K-Means on raw features only. Discovers regimes as regions
    // in X-space, independent of y-magnitude. Recommended for regime-
    // switching problems where the regime is a function of features
    // (e.g. macro indicators, market microstructure) rather than y itself.
    Log::Info("MixtureGBDT: Using Balanced K-Means init on features only "
              "(regime discovery in X-space)");
    InitResponsibilitiesBalancedKMeans(labels, /*include_label=*/false);

  } else if (config_->mixture_init == "gmm") {
    // GMM on (features, label). Soft responsibilities aligned with EM
    // theory (Jacobs 1991), but again y is included as a dim so y-magnitude
    // dominates the partition.
    Log::Info("MixtureGBDT: Using GMM init on features + label");
    InitResponsibilitiesGMM(labels, /*include_label=*/true);

  } else if (config_->mixture_init == "gmm_features") {
    // GMM on raw features only — the cleanest probabilistic regime-init
    // when regimes live in X-space.
    Log::Info("MixtureGBDT: Using GMM init on features only "
              "(regime discovery in X-space)");
    InitResponsibilitiesGMM(labels, /*include_label=*/false);

  } else if (config_->mixture_init == "tree_hierarchical") {
    // Tree-based hierarchical clustering initialization
    // Trains a deep decision tree, then clusters leaves by mean y
    // Reference: Similar to MoEfication's co-activation graph approach
    Log::Info("MixtureGBDT: Using tree-based hierarchical clustering initialization");

    InitResponsibilitiesTreeHierarchical(labels);

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

  // After whichever scheme just ran, ensure r is not pathologically uniform
  // — that's an EM fixed point that empirically left the model frozen for
  // `mixture_init=uniform` even with hard_m_step / variance estimation /
  // diversity_lambda turned up (verified in examples/em_init_sensitivity.py).
  // The breaker is a no-op when r is already non-uniform.
  BreakUniformSymmetryIfNeeded();
}

void MixtureGBDT::InitResponsibilitiesBalancedKMeans(const label_t* labels,
                                                     bool include_label) {
  // Balanced K-Means with optional label dimension.
  // Reference: MoEfication (ACL 2022) uses Balanced K-Means for expert assignment.
  //
  // Algorithm:
  // 1. Initialize centroids using K-means++
  // 2. Iterate: assign samples to nearest centroid
  // 3. Balance: ensure each cluster has exactly N/K samples (greedy)
  //
  // include_label=true:  cluster on (features ⊕ label) — biased toward y
  // include_label=false: cluster on features only — true regime discovery
  //
  // Feature values are read via Dataset::FeatureIterator, which returns the
  // *bin* index for each (feature, sample) pair. The previous implementation
  // used Dataset::raw_index(f), which is null whenever the dataset wasn't
  // created with keep_raw_data=true (the LightGBM default is false). With
  // raw features unavailable, the code emitted a warning and silently fell
  // back to labels-only clustering — the "K-means on features" mode was
  // never actually K-means on features for the typical user. Bin indices
  // are an int discretization of the continuous features (typically 256
  // bins), which is sufficient resolution for K-means seeding of K experts
  // and is available regardless of keep_raw_data. The z-score
  // normalization below absorbs the per-feature scale of the bins.

  const int K = num_experts_;
  const data_size_t N = num_data_;
  const int max_iters = 20;

  const int num_features = train_data_->num_features();
  const int D = num_features + (include_label ? 1 : 0);
  if (D == 0) {
    Log::Warning("MixtureGBDT: Cannot run Balanced K-Means with 0 dimensions, "
                 "falling back to uniform responsibilities");
    const double uniform_r = 1.0 / num_experts_;
    for (data_size_t i = 0; i < N; ++i) {
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] = uniform_r;
      }
    }
    return;
  }

  std::vector<double> X(static_cast<size_t>(N) * D);
  std::vector<double> feat_mean(D, 0.0);
  std::vector<double> feat_std(D, 1.0);

  std::vector<BinIterator*> iters(num_features, nullptr);
  for (int f = 0; f < num_features; ++f) {
    iters[f] = train_data_->FeatureIterator(f);
  }

  for (data_size_t i = 0; i < N; ++i) {
    for (int f = 0; f < num_features; ++f) {
      const double v = static_cast<double>(iters[f]->RawGet(i));
      X[i * D + f] = v;
      feat_mean[f] += v;
    }
    if (include_label) {
      X[i * D + num_features] = static_cast<double>(labels[i]);
      feat_mean[num_features] += X[i * D + num_features];
    }
  }

  for (BinIterator* p : iters) {
    delete p;
  }

  // Compute means
  for (int d = 0; d < D; ++d) {
    feat_mean[d] /= N;
  }

  // Compute standard deviations
  for (data_size_t i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      double diff = X[i * D + d] - feat_mean[d];
      feat_std[d] += diff * diff;
    }
  }
  for (int d = 0; d < D; ++d) {
    feat_std[d] = std::sqrt(feat_std[d] / N);
    if (feat_std[d] < 1e-10) feat_std[d] = 1.0;  // Avoid division by zero
  }

  // Normalize features (z-score)
  for (data_size_t i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      X[i * D + d] = (X[i * D + d] - feat_mean[d]) / feat_std[d];
    }
  }

  // Initialize centroids using K-means++
  std::vector<double> centroids(static_cast<size_t>(K) * D, 0.0);
  std::mt19937 rng(config_->seed);

  // First centroid: random sample
  std::uniform_int_distribution<data_size_t> sample_dist(0, N - 1);
  data_size_t first_idx = sample_dist(rng);
  for (int d = 0; d < D; ++d) {
    centroids[0 * D + d] = X[first_idx * D + d];
  }

  // Remaining centroids: K-means++ (proportional to squared distance)
  std::vector<double> min_dist_sq(N, std::numeric_limits<double>::max());
  for (int k = 1; k < K; ++k) {
    // Update minimum distances to existing centroids
    double total_dist = 0.0;
    for (data_size_t i = 0; i < N; ++i) {
      double dist_sq = 0.0;
      for (int d = 0; d < D; ++d) {
        double diff = X[i * D + d] - centroids[(k - 1) * D + d];
        dist_sq += diff * diff;
      }
      min_dist_sq[i] = std::min(min_dist_sq[i], dist_sq);
      total_dist += min_dist_sq[i];
    }

    // Sample next centroid proportional to squared distance
    std::uniform_real_distribution<double> prob_dist(0.0, total_dist);
    double r = prob_dist(rng);
    double cumsum = 0.0;
    data_size_t next_idx = 0;
    for (data_size_t i = 0; i < N; ++i) {
      cumsum += min_dist_sq[i];
      if (cumsum >= r) {
        next_idx = i;
        break;
      }
    }
    for (int d = 0; d < D; ++d) {
      centroids[k * D + d] = X[next_idx * D + d];
    }
  }

  // K-means iterations
  std::vector<int> assignments(N);
  std::vector<double> distances(static_cast<size_t>(N) * K);

  for (int iter = 0; iter < max_iters; ++iter) {
    // Compute distances to all centroids
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < N; ++i) {
      for (int k = 0; k < K; ++k) {
        double dist_sq = 0.0;
        for (int d = 0; d < D; ++d) {
          double diff = X[i * D + d] - centroids[k * D + d];
          dist_sq += diff * diff;
        }
        distances[i * K + k] = dist_sq;
      }
    }

    // Balanced assignment using greedy approach
    // Target: each cluster gets exactly N/K samples (with remainder distributed)
    const data_size_t base_size = N / K;
    const data_size_t remainder = N % K;
    std::vector<data_size_t> cluster_sizes(K, 0);
    std::vector<data_size_t> cluster_capacity(K);
    for (int k = 0; k < K; ++k) {
      cluster_capacity[k] = base_size + (k < remainder ? 1 : 0);
    }

    // Sort samples by their minimum distance to any centroid (greedy)
    std::vector<std::pair<double, data_size_t>> sample_order(N);
    for (data_size_t i = 0; i < N; ++i) {
      double min_d = distances[i * K];
      for (int k = 1; k < K; ++k) {
        min_d = std::min(min_d, distances[i * K + k]);
      }
      sample_order[i] = {min_d, i};
    }
    std::sort(sample_order.begin(), sample_order.end());

    // Assign samples (closest first)
    std::fill(assignments.begin(), assignments.end(), -1);
    for (const auto& [min_d, i] : sample_order) {
      // Find closest available cluster
      int best_k = -1;
      double best_dist = std::numeric_limits<double>::max();
      for (int k = 0; k < K; ++k) {
        if (cluster_sizes[k] < cluster_capacity[k] && distances[i * K + k] < best_dist) {
          best_dist = distances[i * K + k];
          best_k = k;
        }
      }
      if (best_k >= 0) {
        assignments[i] = best_k;
        cluster_sizes[best_k]++;
      }
    }

    // Handle any unassigned (shouldn't happen with proper capacity)
    for (data_size_t i = 0; i < N; ++i) {
      if (assignments[i] < 0) {
        // Find any cluster with room
        for (int k = 0; k < K; ++k) {
          if (cluster_sizes[k] < cluster_capacity[k]) {
            assignments[i] = k;
            cluster_sizes[k]++;
            break;
          }
        }
      }
    }

    // Update centroids
    std::vector<double> new_centroids(static_cast<size_t>(K) * D, 0.0);
    std::vector<int> counts(K, 0);

    for (data_size_t i = 0; i < N; ++i) {
      int k = assignments[i];
      for (int d = 0; d < D; ++d) {
        new_centroids[k * D + d] += X[i * D + d];
      }
      counts[k]++;
    }

    for (int k = 0; k < K; ++k) {
      if (counts[k] > 0) {
        for (int d = 0; d < D; ++d) {
          new_centroids[k * D + d] /= counts[k];
        }
      }
    }

    // Check convergence
    double max_shift = 0.0;
    for (int k = 0; k < K; ++k) {
      for (int d = 0; d < D; ++d) {
        max_shift = std::max(max_shift,
                             std::abs(new_centroids[k * D + d] - centroids[k * D + d]));
      }
    }
    centroids = std::move(new_centroids);

    if (max_shift < 1e-6) {
      Log::Debug("MixtureGBDT: Balanced K-Means converged at iteration %d", iter + 1);
      break;
    }
  }

  // Convert assignments to soft responsibilities
  const double base_r = 0.05 / K;  // Small base probability
  const double main_r = 1.0 - base_r * K;  // Main probability for assigned expert

  for (data_size_t i = 0; i < N; ++i) {
    int assigned_k = assignments[i];
    for (int k = 0; k < K; ++k) {
      if (k == assigned_k) {
        responsibilities_[i * num_experts_ + k] = main_r + base_r;
      } else {
        responsibilities_[i * num_experts_ + k] = base_r;
      }
    }
  }

  // Log cluster sizes for verification
  std::vector<int> final_counts(K, 0);
  for (data_size_t i = 0; i < N; ++i) {
    final_counts[assignments[i]]++;
  }
  std::string count_str;
  for (int k = 0; k < K; ++k) {
    count_str += std::to_string(final_counts[k]);
    if (k < K - 1) count_str += ", ";
  }
  Log::Info("MixtureGBDT: Balanced K-Means cluster sizes = [%s]", count_str.c_str());
}

void MixtureGBDT::InitResponsibilitiesGMM(const label_t* labels,
                                          bool include_label) {
  // Gaussian Mixture Model initialization
  // Reference: Classical MoE (Jacobs et al., 1991)
  //
  // Algorithm:
  // 1. Initialize with K-means centroids
  // 2. EM iterations:
  //    E-step: compute posterior probabilities (responsibilities)
  //    M-step: update means, variances, mixing coefficients
  // 3. Final posteriors become the initial responsibilities
  //
  // Feature values are read via Dataset::FeatureIterator (bin indices), the
  // same fix used for InitResponsibilitiesBalancedKMeans — see that
  // function for the rationale on why raw_index() was wrong by default.

  const int K = num_experts_;
  const data_size_t N = num_data_;
  const int max_iters = 30;  // EM iterations
  const double min_variance = 1e-6;  // Prevent collapse

  const int num_features = train_data_->num_features();
  const int D = num_features + (include_label ? 1 : 0);
  if (D == 0) {
    Log::Warning("MixtureGBDT: Cannot run GMM with 0 dimensions, falling "
                 "back to uniform responsibilities");
    const double uniform_r = 1.0 / num_experts_;
    for (data_size_t i = 0; i < N; ++i) {
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] = uniform_r;
      }
    }
    return;
  }

  std::vector<double> X(static_cast<size_t>(N) * D);

  std::vector<BinIterator*> iters(num_features, nullptr);
  for (int f = 0; f < num_features; ++f) {
    iters[f] = train_data_->FeatureIterator(f);
  }

  for (data_size_t i = 0; i < N; ++i) {
    for (int f = 0; f < num_features; ++f) {
      X[i * D + f] = static_cast<double>(iters[f]->RawGet(i));
    }
    if (include_label) {
      X[i * D + num_features] = static_cast<double>(labels[i]);
    }
  }

  for (BinIterator* p : iters) {
    delete p;
  }

  // Compute global mean and std for normalization
  std::vector<double> global_mean(D, 0.0);
  std::vector<double> global_std(D, 1.0);

  for (data_size_t i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      global_mean[d] += X[i * D + d];
    }
  }
  for (int d = 0; d < D; ++d) {
    global_mean[d] /= N;
  }
  for (data_size_t i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      double diff = X[i * D + d] - global_mean[d];
      global_std[d] += diff * diff;
    }
  }
  for (int d = 0; d < D; ++d) {
    global_std[d] = std::sqrt(global_std[d] / N);
    if (global_std[d] < 1e-10) global_std[d] = 1.0;
  }

  // Normalize
  for (data_size_t i = 0; i < N; ++i) {
    for (int d = 0; d < D; ++d) {
      X[i * D + d] = (X[i * D + d] - global_mean[d]) / global_std[d];
    }
  }

  // Initialize GMM parameters using quantile-based seeding
  // (sort by first feature or label, then place centroids at quantile positions)
  std::vector<double> means(static_cast<size_t>(K) * D);      // K x D
  std::vector<double> variances(static_cast<size_t>(K) * D);  // K x D (diagonal covariance)
  std::vector<double> weights(K, 1.0 / K);                    // Mixing coefficients

  // Sort indices by last feature (label)
  std::vector<data_size_t> sorted_idx(N);
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::sort(sorted_idx.begin(), sorted_idx.end(),
            [&X, D](data_size_t a, data_size_t b) {
              return X[a * D + D - 1] < X[b * D + D - 1];
            });

  // Place initial means at quantile positions
  for (int k = 0; k < K; ++k) {
    data_size_t quantile_idx = sorted_idx[(k * N + N / 2) / K];
    for (int d = 0; d < D; ++d) {
      means[k * D + d] = X[quantile_idx * D + d];
      variances[k * D + d] = 1.0;  // Initial variance = 1 (normalized data)
    }
  }

  // EM iterations
  std::vector<double> gamma(static_cast<size_t>(N) * K);  // Responsibilities

  for (int iter = 0; iter < max_iters; ++iter) {
    // E-step: compute responsibilities
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < N; ++i) {
      std::vector<double> log_prob(K);
      double max_log_prob = -std::numeric_limits<double>::max();

      for (int k = 0; k < K; ++k) {
        // Log probability under Gaussian k (diagonal covariance)
        double log_p = std::log(weights[k] + 1e-300);
        for (int d = 0; d < D; ++d) {
          double var_k = std::max(variances[k * D + d], min_variance);
          double diff = X[i * D + d] - means[k * D + d];
          log_p -= 0.5 * std::log(2.0 * M_PI * var_k);
          log_p -= 0.5 * diff * diff / var_k;
        }
        log_prob[k] = log_p;
        max_log_prob = std::max(max_log_prob, log_p);
      }

      // Softmax to get responsibilities
      double sum_exp = 0.0;
      for (int k = 0; k < K; ++k) {
        gamma[i * K + k] = std::exp(log_prob[k] - max_log_prob);
        sum_exp += gamma[i * K + k];
      }
      for (int k = 0; k < K; ++k) {
        gamma[i * K + k] /= sum_exp;
      }
    }

    // M-step: update parameters
    std::vector<double> N_k(K, 0.0);  // Effective number of points per cluster
    std::vector<double> new_means(static_cast<size_t>(K) * D, 0.0);
    std::vector<double> new_variances(static_cast<size_t>(K) * D, 0.0);

    // Compute new means
    for (data_size_t i = 0; i < N; ++i) {
      for (int k = 0; k < K; ++k) {
        double g = gamma[i * K + k];
        N_k[k] += g;
        for (int d = 0; d < D; ++d) {
          new_means[k * D + d] += g * X[i * D + d];
        }
      }
    }
    for (int k = 0; k < K; ++k) {
      if (N_k[k] > 1e-10) {
        for (int d = 0; d < D; ++d) {
          new_means[k * D + d] /= N_k[k];
        }
      }
    }

    // Compute new variances
    for (data_size_t i = 0; i < N; ++i) {
      for (int k = 0; k < K; ++k) {
        double g = gamma[i * K + k];
        for (int d = 0; d < D; ++d) {
          double diff = X[i * D + d] - new_means[k * D + d];
          new_variances[k * D + d] += g * diff * diff;
        }
      }
    }
    for (int k = 0; k < K; ++k) {
      if (N_k[k] > 1e-10) {
        for (int d = 0; d < D; ++d) {
          new_variances[k * D + d] /= N_k[k];
          new_variances[k * D + d] = std::max(new_variances[k * D + d], min_variance);
        }
      }
    }

    // Update weights
    for (int k = 0; k < K; ++k) {
      weights[k] = N_k[k] / N;
      weights[k] = std::max(weights[k], 1e-10);  // Prevent zero weight
    }
    // Renormalize weights
    double weight_sum = 0.0;
    for (int k = 0; k < K; ++k) weight_sum += weights[k];
    for (int k = 0; k < K; ++k) weights[k] /= weight_sum;

    // Check convergence (mean shift)
    double max_shift = 0.0;
    for (int k = 0; k < K; ++k) {
      for (int d = 0; d < D; ++d) {
        max_shift = std::max(max_shift, std::abs(new_means[k * D + d] - means[k * D + d]));
      }
    }

    means = std::move(new_means);
    variances = std::move(new_variances);

    if (max_shift < 1e-6) {
      Log::Debug("MixtureGBDT: GMM converged at iteration %d", iter + 1);
      break;
    }
  }

  // Copy final responsibilities (gamma)
  // Note: GMM gives soft assignments naturally
  for (data_size_t i = 0; i < N; ++i) {
    for (int k = 0; k < K; ++k) {
      // Ensure minimum responsibility
      responsibilities_[i * num_experts_ + k] = std::max(gamma[i * K + k], 0.01 / K);
    }
    // Renormalize
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += responsibilities_[i * num_experts_ + k];
    }
    for (int k = 0; k < K; ++k) {
      responsibilities_[i * num_experts_ + k] /= sum;
    }
  }

  // Log weight distribution
  std::string weight_str;
  for (int k = 0; k < K; ++k) {
    weight_str += std::to_string(weights[k]).substr(0, 5);
    if (k < K - 1) weight_str += ", ";
  }
  Log::Info("MixtureGBDT: GMM mixing weights = [%s]", weight_str.c_str());
}

void MixtureGBDT::InitResponsibilitiesTreeHierarchical(const label_t* labels) {
  // Hierarchical clustering initialization based on label distribution
  //
  // Algorithm:
  // 1. Create fine-grained bins based on label values (like tree leaves)
  // 2. Compute mean y and count for each bin
  // 3. Build distance matrix between bins (|mean_y_i - mean_y_j|)
  // 4. Agglomerative hierarchical clustering (average linkage) to merge bins into K groups
  // 5. Assign samples to experts based on their bin's cluster
  //
  // This provides a principled way to partition the label space into K regions
  // that are as homogeneous as possible within each cluster.

  const int K = num_experts_;
  const data_size_t N = num_data_;

  // Step 1: Create fine-grained bins based on label values
  // Use more bins than experts for fine-grained initial partitioning
  const int num_bins = std::max(64, K * 16);

  // Sort samples by label to find bin boundaries
  std::vector<data_size_t> sorted_indices(N);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [labels](data_size_t a, data_size_t b) { return labels[a] < labels[b]; });

  // Assign samples to bins (equal-frequency binning)
  std::vector<int> bin_assignment(N);
  std::vector<std::vector<data_size_t>> bin_samples(num_bins);
  std::vector<double> bin_mean_y(num_bins, 0.0);
  std::vector<int> bin_count(num_bins, 0);

  for (data_size_t rank = 0; rank < N; ++rank) {
    data_size_t i = sorted_indices[rank];
    int bin = static_cast<int>(static_cast<int64_t>(rank) * num_bins / N);
    if (bin >= num_bins) bin = num_bins - 1;
    bin_assignment[i] = bin;
    bin_samples[bin].push_back(i);
    bin_mean_y[bin] += labels[i];
    bin_count[bin]++;
  }

  // Compute mean y for each bin
  for (int b = 0; b < num_bins; ++b) {
    if (bin_count[b] > 0) {
      bin_mean_y[b] /= bin_count[b];
    }
  }

  Log::Info("MixtureGBDT: Created %d bins for hierarchical clustering", num_bins);

  // Step 2: Hierarchical clustering (agglomerative with average linkage)
  // We'll merge bins until we have K clusters

  if (num_bins <= K) {
    // If we have fewer bins than experts, assign each bin to a different expert
    Log::Warning("MixtureGBDT: Only %d bins, less than K=%d experts", num_bins, K);

    // Sort bins by mean y and distribute
    std::vector<std::pair<double, int>> bin_order;
    for (int b = 0; b < num_bins; ++b) {
      bin_order.push_back({bin_mean_y[b], b});
    }
    std::sort(bin_order.begin(), bin_order.end());

    std::vector<int> bin_to_expert(num_bins);
    for (size_t i = 0; i < bin_order.size(); ++i) {
      int expert = static_cast<int>(i * K / num_bins);
      if (expert >= K) expert = K - 1;
      bin_to_expert[bin_order[i].second] = expert;
    }

    // Assign responsibilities
    const double base_r = 0.05 / K;
    const double main_r = 1.0 - base_r * K;

    for (data_size_t i = 0; i < N; ++i) {
      int bin = bin_assignment[i];
      int assigned_expert = bin_to_expert[bin];
      for (int k = 0; k < K; ++k) {
        responsibilities_[i * num_experts_ + k] = (k == assigned_expert) ? main_r + base_r : base_r;
      }
    }
  } else {
    // Hierarchical clustering to merge bins into K clusters
    // Using agglomerative clustering with average linkage

    // Initialize: each bin is its own cluster
    // cluster_members[c] = list of bin indices in cluster c
    std::vector<std::vector<int>> cluster_members(num_bins);
    for (int b = 0; b < num_bins; ++b) {
      cluster_members[b].push_back(b);
    }

    // Compute initial cluster centroids (mean y of cluster)
    std::vector<double> cluster_centroid(num_bins);
    std::vector<int> cluster_weight(num_bins);  // Total sample count in cluster
    for (int b = 0; b < num_bins; ++b) {
      cluster_centroid[b] = bin_mean_y[b];
      cluster_weight[b] = bin_count[b];
    }

    // Track active clusters
    std::vector<bool> active(num_bins, true);
    int num_active = num_bins;

    // Merge until K clusters remain
    while (num_active > K) {
      // Find closest pair of clusters
      double min_dist = std::numeric_limits<double>::max();
      int merge_i = -1, merge_j = -1;

      for (int i = 0; i < num_bins; ++i) {
        if (!active[i]) continue;
        for (int j = i + 1; j < num_bins; ++j) {
          if (!active[j]) continue;

          // Distance = |centroid_i - centroid_j|
          double dist = std::abs(cluster_centroid[i] - cluster_centroid[j]);

          if (dist < min_dist) {
            min_dist = dist;
            merge_i = i;
            merge_j = j;
          }
        }
      }

      if (merge_i < 0 || merge_j < 0) break;

      // Merge cluster j into cluster i
      // Update centroid using weighted average
      double new_weight = cluster_weight[merge_i] + cluster_weight[merge_j];
      cluster_centroid[merge_i] = (cluster_centroid[merge_i] * cluster_weight[merge_i] +
                                    cluster_centroid[merge_j] * cluster_weight[merge_j]) / new_weight;
      cluster_weight[merge_i] = static_cast<int>(new_weight);

      // Move members from j to i
      for (int bin_idx : cluster_members[merge_j]) {
        cluster_members[merge_i].push_back(bin_idx);
      }
      cluster_members[merge_j].clear();

      // Deactivate cluster j
      active[merge_j] = false;
      num_active--;
    }

    // Create bin -> expert mapping
    std::vector<int> bin_to_expert(num_bins);
    int expert_idx = 0;

    // Sort clusters by centroid for consistent ordering
    std::vector<std::pair<double, int>> cluster_order;
    for (int i = 0; i < num_bins; ++i) {
      if (active[i]) {
        cluster_order.push_back({cluster_centroid[i], i});
      }
    }
    std::sort(cluster_order.begin(), cluster_order.end());

    for (const auto& [centroid, cluster_idx] : cluster_order) {
      for (int bin_local_idx : cluster_members[cluster_idx]) {
        bin_to_expert[bin_local_idx] = expert_idx;
      }
      expert_idx++;
    }

    // Assign responsibilities
    const double base_r = 0.05 / K;
    const double main_r = 1.0 - base_r * K;

    for (data_size_t i = 0; i < N; ++i) {
      int bin = bin_assignment[i];
      int assigned_expert = bin_to_expert[bin];
      for (int k = 0; k < K; ++k) {
        responsibilities_[i * num_experts_ + k] = (k == assigned_expert) ? main_r + base_r : base_r;
      }
    }

    // Log cluster sizes
    std::vector<int> expert_counts(K, 0);
    for (data_size_t i = 0; i < N; ++i) {
      int bin = bin_assignment[i];
      expert_counts[bin_to_expert[bin]]++;
    }

    std::string count_str;
    for (int k = 0; k < K; ++k) {
      count_str += std::to_string(expert_counts[k]);
      if (k < K - 1) count_str += ", ";
    }
    Log::Info("MixtureGBDT: Tree hierarchical cluster sizes = [%s]", count_str.c_str());
  }
}

void MixtureGBDT::ComputeGateProbForInference(const double* gate_raw,
                                              double* gate_prob) const {
  // Mirror Forward()/ForwardValid(): apply per-expert bias and temperature
  // before softmax. Without this, models trained with non-default
  // `mixture_balance_factor` (which drives expert_bias_) or temperature
  // annealing produce a routing at inference that does not match the routing
  // used during training — silently degrading test metrics.
  std::vector<double> scores(num_experts_);
  const double inv_T = 1.0 / std::max(gate_temperature_, kMixtureEpsilon);
  for (int k = 0; k < num_experts_; ++k) {
    scores[k] = (gate_raw[k] + expert_bias_[k]) * inv_T;
  }
  Softmax(scores.data(), num_experts_, gate_prob);
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
    // Pull the quantile level τ from config (same field LightGBM's quantile
    // objective uses). Earlier this was hardcoded to 0.5 with a TODO, so
    // E-step responsibilities were computed against the median even when the
    // user trained for τ=0.9 — silently inconsistent with the objective.
    const double tau = config_->alpha;
    if (diff >= 0) {
      return tau * diff;
    } else {
      return (tau - 1.0) * diff;
    }
  }
  // Default to L2
  return diff * diff;
}

double MixtureGBDT::ComputeTemperature(int moe_iter, int total_moe_iters) const {
  double t_init = config_->mixture_gate_temperature_init;
  double t_final = config_->mixture_gate_temperature_final;
  if (t_init == t_final) return t_init;  // No annealing

  // Exponential decay: T(t) = T_init * (T_final/T_init)^(t/T_total)
  double progress = std::min(1.0, static_cast<double>(moe_iter) / std::max(1, total_moe_iters));
  return t_init * std::pow(t_final / t_init, progress);
}

void MixtureGBDT::BreakUniformSymmetryIfNeeded() {
  // Detect: every responsibility row is essentially uniform (within tol).
  // Catches `mixture_init=uniform` and any pathological init that happens to
  // land near uniform (rare but possible with degenerate features).
  const double uniform = 1.0 / num_experts_;
  const double tol = 1e-6;
  bool is_uniform = true;
  for (data_size_t i = 0; i < num_data_ && is_uniform; ++i) {
    for (int k = 0; k < num_experts_; ++k) {
      if (std::abs(responsibilities_[i * num_experts_ + k] - uniform) > tol) {
        is_uniform = false;
        break;
      }
    }
  }
  if (!is_uniform) return;

  // Inject a deterministic, expert-distinct, sample-varying perturbation.
  // Sinusoidal because it's bounded, smooth, and gives every (i, k) pair a
  // different displacement (no two experts collide on any sample). Magnitude
  // ε=0.05 is small enough to remain a valid stochastic distribution after
  // renormalization but large enough that downstream gradients see different
  // weights per expert — the only thing the EM fixed-point loop required.
  const double epsilon = 0.05;
  const double n_inv = 2.0 * M_PI / std::max<data_size_t>(num_data_, 1);
  for (data_size_t i = 0; i < num_data_; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      const double phase = n_inv * static_cast<double>(i) * (k + 1);
      const double v = std::max(0.0, uniform + epsilon * std::sin(phase));
      responsibilities_[i * num_experts_ + k] = v;
      sum += v;
    }
    const double inv_sum = 1.0 / std::max(sum, kMixtureEpsilon);
    for (int k = 0; k < num_experts_; ++k) {
      responsibilities_[i * num_experts_ + k] *= inv_sum;
    }
  }
  Log::Info("MixtureGBDT: detected uniform r_init; applied deterministic "
            "symmetry-breaking perturbation (eps=%.2f). Without this, EM is "
            "stuck at the uniform fixed point — verified empirically in "
            "examples/em_init_sensitivity.py.",
            epsilon);
}

void MixtureGBDT::SpawnExpertsFromSeed() {
  const double perturbation = config_->mixture_spawn_perturbation;
  Log::Info("MixtureGBDT: Spawning %d experts from seed "
            "(perturbation=%.2f, seed has %d trees)",
            num_experts_, perturbation, seed_expert_->NumberOfTotalModel());

  experts_.clear();
  experts_.reserve(num_experts_);

  for (int k = 0; k < num_experts_; ++k) {
    experts_.emplace_back(new GBDT());
    // Init creates full training infrastructure (tree_learner, score_updater, etc.)
    experts_[k]->Init(expert_configs_[k].get(), train_data_, nullptr, {});

    for (size_t v = 0; v < valid_datas_.size(); ++v) {
      experts_[k]->AddValidDataset(valid_datas_[v], {});
    }

    // Merge seed trees into this expert (copies all trees, sets num_init_iteration_)
    experts_[k]->MergeFrom(seed_expert_.get());

    // Sync score updaters with merged trees so GetPredictAt returns correct values
    experts_[k]->SyncScoresFromMergedTrees();

    // Apply perturbation: randomly perturb leaf values to break symmetry
    if (perturbation > 0.0 && k > 0) {
      // Expert 0 keeps the exact seed copy; others get perturbed
      const int num_trees = seed_expert_->NumberOfTotalModel();
      std::mt19937 rng(config_->seed + k + 10000);
      std::uniform_real_distribution<double> drop_dist(0.0, 1.0);
      std::normal_distribution<double> noise_dist(0.0, 1.0);

      for (int t = 0; t < num_trees; ++t) {
        if (drop_dist(rng) < perturbation) {
          // Perturb this tree's leaf values
          const int num_leaves = experts_[k]->GetNumLeavesForTree(t);
          for (int l = 0; l < num_leaves; ++l) {
            double orig = experts_[k]->GetLeafValue(t, l);
            double noise_scale = std::max(std::abs(orig), 0.01);
            double new_val = orig + noise_scale * noise_dist(rng) * 0.5;
            experts_[k]->SetLeafValue(t, l, new_val);
          }
        }
      }
    }
  }

  Log::Info("MixtureGBDT: Experts spawned from seed successfully");

  // Free seed expert
  seed_expert_.reset();
}

void MixtureGBDT::Forward() {
  // Get expert predictions
  for (int k = 0; k < num_experts_; ++k) {
    int64_t out_len;
    experts_[k]->GetPredictAt(0, expert_pred_.data() + k * num_data_, &out_len);
  }

  // Get gate probabilities
  if (config_->mixture_gate_type == "none" || config_->mixture_gate_type == "leaf_reuse") {
    // "none": use previous responsibilities as routing probabilities
    // "leaf_reuse": gate_proba_ was set by MStepGate's leaf statistics (keep as-is)
    // On first iteration, responsibilities are from InitResponsibilities
    if (config_->mixture_gate_type == "none") {
      std::copy(responsibilities_.begin(), responsibilities_.end(), gate_proba_.begin());
    }
    // For leaf_reuse, gate_proba_ is already set from previous MStepGate call.
    // Neither mode applies expert_bias_ to its routing — the bias-free view
    // therefore equals the routing distribution exactly.
    std::copy(gate_proba_.begin(), gate_proba_.end(), gate_proba_no_bias_.begin());
  } else {
    // GBDT gate: softmax of gate raw predictions.
    //
    // Compute TWO views of the routing distribution per sample:
    //   gate_proba_no_bias_ = softmax(z / T)           ← prior for E-step / ELBO
    //   gate_proba_         = softmax((z + b) / T)     ← actual routing for yhat
    //
    // Splitting these matters for DeepSeek "Auxiliary-Loss-Free Load
    // Balancing" semantics: the bias is a routing-decision nudge; it must
    // NOT enter the probabilistic prior that defines responsibilities, or
    // the gate would have to spend each iter undoing the bias the load
    // balancer just added (PR #25 fixed this on the gradient side via
    // bias-free p; the prior side stayed bias-tainted until this fix).
    std::vector<double> gate_raw(static_cast<size_t>(num_data_) * num_experts_);
    int64_t out_len;
    gate_->GetPredictAt(0, gate_raw.data(), &out_len);

    const double inv_T = 1.0 / std::max(gate_temperature_, kMixtureEpsilon);
    // gate_raw is in class-major order: gate_raw[k * num_data_ + i] = score for sample i, class k
    // gate_proba_ / gate_proba_no_bias_ are in sample-major order
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      std::vector<double> scores_no_bias(num_experts_);
      std::vector<double> scores_with_bias(num_experts_);
      for (int k = 0; k < num_experts_; ++k) {
        const double z_over_T = gate_raw[k * num_data_ + i] * inv_T;
        scores_no_bias[k]   = z_over_T;
        scores_with_bias[k] = z_over_T + expert_bias_[k] * inv_T;
      }
      Softmax(scores_no_bias.data(), num_experts_,
              gate_proba_no_bias_.data() + i * num_experts_);
      Softmax(scores_with_bias.data(), num_experts_,
              gate_proba_.data() + i * num_experts_);
    }
  }

  // Markov mode: temporal smoothing of routing along the row (time) axis.
  //
  // Audit fix: previously this used a class-member `prev_gate_proba_` that
  // got overwritten with gate_proba_[i-1] after each iteration's blend, then
  // re-used as the smoothing source on the *next* training iteration. That
  // accumulated an iteration-axis EMA on top of the time-axis shift, so
  // sample i's "previous" was an exponentially weighted average of past
  // training iterations' (already-smoothed) routing — not a Markov prior.
  //
  // The corrected smoothing is a single-pass forward sweep using only the
  // unsmoothed value of row i-1 from THIS iteration as sample i's prior. No
  // state survives across training iterations.
  //
  // We sweep BOTH gate_proba_ and gate_proba_no_bias_: Markov smoothing is a
  // probabilistic model assumption about the time evolution of routing, not
  // a routing-decision nudge — so it applies symmetrically to the routing
  // (used by yhat) and to the prior (used by E-step / ELBO).
  if (use_markov_) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0 && num_data_ > 1) {
      auto markov_sweep = [&](double* buf) {
        std::vector<double> prev_row(num_experts_);
        // Row 0 is never blended (no prior available).
        for (int k = 0; k < num_experts_; ++k) {
          prev_row[k] = buf[k];
        }
        for (data_size_t i = 1; i < num_data_; ++i) {
          // Snapshot row i's unsmoothed value before the blend.
          std::vector<double> cur_row(num_experts_);
          for (int k = 0; k < num_experts_; ++k) {
            cur_row[k] = buf[i * num_experts_ + k];
          }
          double sum = 0.0;
          for (int k = 0; k < num_experts_; ++k) {
            buf[i * num_experts_ + k] =
                (1.0 - lambda) * cur_row[k] + lambda * prev_row[k];
            sum += buf[i * num_experts_ + k];
          }
          // Renormalize (numerical drift only; both inputs were already
          // probability vectors).
          const double inv_sum = 1.0 / std::max(sum, kMixtureEpsilon);
          for (int k = 0; k < num_experts_; ++k) {
            buf[i * num_experts_ + k] *= inv_sum;
          }
          // Advance prev_row to row i's UNSMOOTHED value (so row i+1 gets a
          // clean Markov prior, not a doubly-smoothed one).
          prev_row.swap(cur_row);
        }
      };
      markov_sweep(gate_proba_.data());
      markov_sweep(gate_proba_no_bias_.data());
    }
    // prev_gate_proba_ is no longer carried across iterations; it remains
    // sized for back-compat with the predict-time PredictWithPrevProba path
    // which takes its prior from a caller-supplied argument anyway.
  }

  // Transpose expert_pred_ (expert-major) → expert_pred_sm_ (sample-major)
  // so that E-step and yhat can read [i*K + k] with sequential cache access
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    for (int k = 0; k < num_experts_; ++k) {
      expert_pred_sm_[i * num_experts_ + k] = expert_pred_[k * num_data_ + i];
    }
  }

  // Compute combined prediction: yhat[i] = sum_k gate_proba[i,k] * expert_pred_sm[i,k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    double sum = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      sum += gate_proba_[i * num_experts_ + k] * expert_pred_sm_[i * num_experts_ + k];
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

  // Apply softmax per sample with expert bias and temperature scaling
  // gate_raw is in class-major order: gate_raw[k * num_valid + i]
  // gate_proba is in sample-major order: gate_proba[i * num_experts_ + k]
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_valid; ++i) {
    std::vector<double> scores(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      scores[k] = (gate_raw[k * num_valid + i] + expert_bias_[k]) / gate_temperature_;
    }
    Softmax(scores.data(), num_experts_, gate_proba.data() + i * num_experts_);
  }

  // Markov mode: same single-pass forward sweep as Forward(); see audit
  // note there. No iteration-axis state is carried in prev_gate_proba_valid_;
  // each call computes the time-axis Markov prior fresh from this iter's
  // gate_proba.
  if (use_markov_) {
    const double lambda = config_->mixture_smoothing_lambda;
    if (lambda > 0.0 && num_valid > 1) {
      std::vector<double> prev_row(num_experts_);
      for (int k = 0; k < num_experts_; ++k) {
        prev_row[k] = gate_proba[k];
      }
      for (data_size_t i = 1; i < num_valid; ++i) {
        std::vector<double> cur_row(num_experts_);
        for (int k = 0; k < num_experts_; ++k) {
          cur_row[k] = gate_proba[i * num_experts_ + k];
        }
        double sum = 0.0;
        for (int k = 0; k < num_experts_; ++k) {
          gate_proba[i * num_experts_ + k] =
              (1.0 - lambda) * cur_row[k] + lambda * prev_row[k];
          sum += gate_proba[i * num_experts_ + k];
        }
        const double inv_sum = 1.0 / std::max(sum, kMixtureEpsilon);
        for (int k = 0; k < num_experts_; ++k) {
          gate_proba[i * num_experts_ + k] *= inv_sum;
        }
        prev_row.swap(cur_row);
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
  // NOTE: Token Choice routing is prone to Expert Collapse without load balancing.
  // Use mixture_load_balance_alpha > 0 to enable auxiliary load balancing loss.
  // Alternatively, use Expert Choice routing (mixture_routing_mode="expert_choice").
  const label_t* labels = train_data_->metadata().label();
  const double alpha = config_->mixture_e_step_alpha;
  const double lb_alpha = config_->mixture_load_balance_alpha;
  const bool estimate_var = config_->mixture_estimate_variance;
  // When gate_type="none", force loss_only mode (no gate probabilities available)
  // leaf_reuse has valid gate_proba from leaf statistics, so use configured mode
  const std::string mode = (config_->mixture_gate_type == "none")
      ? "loss_only" : config_->mixture_e_step_mode;

  // Precompute load balance penalty for each expert
  // penalty_k = lb_alpha * log(load_k * K)
  // When load_k > 1/K (overloaded): penalty > 0, score decreases
  // When load_k < 1/K (underloaded): penalty < 0, score increases
  std::vector<double> load_penalty(num_experts_, 0.0);
  if (lb_alpha > 0.0) {
    for (int k = 0; k < num_experts_; ++k) {
      // Add small epsilon to avoid log(0)
      double adjusted_load = std::max(expert_load_[k], 1e-10) * num_experts_;
      load_penalty[k] = lb_alpha * std::log(adjusted_load);
    }
  }

  // Precompute the per-expert log-density normalizer that does NOT depend on i.
  //   Gaussian (l2): log p(y|x,f,σ²) = -0.5 log(2πσ²) - (y-f)²/(2σ²)
  //                  → norm_k = -0.5 log(2π σ_k²)         and  scale_k = 1/(2σ_k²)
  //   Laplace (l1):  log p(y|x,f,b)  = -log(2b) - |y-f|/b
  //                  → norm_k = -log(2 b_k)                and  scale_k = 1/b_k
  //   quantile/other: no proper density. Treat the loss as a pseudo-energy
  //                   using a single scale, no normalizer (cancels in softmax).
  std::vector<double> log_norm(num_experts_, 0.0);
  std::vector<double> inv_scale(num_experts_, alpha);
  if (estimate_var) {
    for (int k = 0; k < num_experts_; ++k) {
      const double s = std::max(expert_variance_[k], kMixtureEpsilon);
      if (e_step_loss_type_ == "l1") {
        // s holds Laplace b_k.
        log_norm[k]  = -std::log(2.0 * s);
        inv_scale[k] = 1.0 / s;
      } else if (e_step_loss_type_ == "l2") {
        // s holds variance σ_k². Constant 0.5*log(2π) is sample-independent
        // and cancels under softmax across k *only if* it doesn't depend on
        // k — which it doesn't, so we drop it for cleanliness. The
        // -0.5*log(σ_k²) term DOES depend on k and is essential.
        log_norm[k]  = -0.5 * std::log(s);
        inv_scale[k] = 1.0 / (2.0 * s);
      } else {
        // quantile / other: 1/scale acts as alpha; no normalizer.
        log_norm[k]  = 0.0;
        inv_scale[k] = 1.0 / s;
      }
    }
  }

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    std::vector<double> scores(num_experts_);

    for (int k = 0; k < num_experts_; ++k) {
      double score = 0.0;

      if (mode == "gate_only") {
        // Prior π_k(x) = bias-free softmax of gate logits. The load-balance
        // bias is a routing nudge (DeepSeek), not a probabilistic prior;
        // mixing it into the responsibility softmax forces the gate to
        // learn to undo bias each iter — defeats the point.
        const double gate_prob = gate_proba_no_bias_[i * num_experts_ + k];
        score = std::log(gate_prob + kMixtureEpsilon);
      } else if (mode == "loss_only") {
        const double expert_p = expert_pred_sm_[i * num_experts_ + k];
        const double loss = ComputePointwiseLoss(labels[i], expert_p);
        if (estimate_var) {
          score = log_norm[k] - inv_scale[k] * loss;
        } else {
          score = -alpha * loss;
        }
      } else {
        // em mode: log π_k(x) + log p(y | x, f_k, scale_k); π_k is bias-free
        // (see gate_only branch above for rationale).
        const double gate_prob = gate_proba_no_bias_[i * num_experts_ + k];
        const double expert_p = expert_pred_sm_[i * num_experts_ + k];
        const double loss = ComputePointwiseLoss(labels[i], expert_p);
        if (estimate_var) {
          score = std::log(gate_prob + kMixtureEpsilon)
                + log_norm[k] - inv_scale[k] * loss;
        } else {
          score = std::log(gate_prob + kMixtureEpsilon) - alpha * loss;
        }
      }

      scores[k] = score - load_penalty[k];
    }

    Softmax(scores.data(), num_experts_, responsibilities_.data() + i * num_experts_);
  }
}

void MixtureGBDT::UpdateExpertVariances() {
  if (!config_->mixture_estimate_variance) return;

  const label_t* labels = train_data_->metadata().label();
  // Standard MoE M-step for the noise scale (per Jordan-Jacobs):
  //   σ_k² = Σ_i r_ik (y_i - f_k(x_i))² / Σ_i r_ik
  //   b_k  = Σ_i r_ik |y_i - f_k(x_i)| / Σ_i r_ik       (Laplace)
  // Using a sample-major reduction with per-thread accumulators to avoid the
  // false-sharing pitfall when num_experts_ is small.
  std::vector<double> num_acc(num_experts_, 0.0);
  std::vector<double> den_acc(num_experts_, 0.0);

  for (data_size_t i = 0; i < num_data_; ++i) {
    const double y = static_cast<double>(labels[i]);
    for (int k = 0; k < num_experts_; ++k) {
      const double r = responsibilities_[i * num_experts_ + k];
      const double f = expert_pred_sm_[i * num_experts_ + k];
      const double diff = y - f;
      // Per loss type, accumulate the residual term whose units match the
      // "scale" the E-step then divides by — i.e. the unitless ratio
      //   loss / scale  must equal the log-density exponent.
      //   l2:        scale = σ² = E[diff²],            E-step uses diff²/(2σ²)
      //   l1:        scale = b  = E[|diff|],           E-step uses |diff|/b
      //   quantile:  scale = E[pinball],               E-step uses pinball/scale
      // The earlier code fell into (diff*diff) for *anything other than l1*,
      // so quantile users got `pinball / E[diff²]` — dimensionally O(1/diff)
      // and silently temperature-coupled to the y-scale.
      double residual_term;
      if (e_step_loss_type_ == "l1") {
        residual_term = std::fabs(diff);
      } else if (e_step_loss_type_ == "quantile") {
        residual_term = ComputePointwiseLoss(y, f);  // pinball loss
      } else {
        residual_term = diff * diff;
      }
      num_acc[k] += r * residual_term;
      den_acc[k] += r;
    }
  }

  for (int k = 0; k < num_experts_; ++k) {
    if (den_acc[k] > kMixtureEpsilon) {
      expert_variance_[k] = std::max(num_acc[k] / den_acc[k], kMixtureEpsilon);
    }
    // else: keep previous estimate; den ~ 0 means no samples are routed to k
  }

  if (iter_ % 10 == 0) {
    std::string buf;
    for (int k = 0; k < num_experts_; ++k) {
      buf += std::to_string(expert_variance_[k]).substr(0, 6) + " ";
    }
    Log::Debug("MixtureGBDT: per-expert noise scale = [%s]", buf.c_str());
  }
}

double MixtureGBDT::ComputeMarginalLogLikelihood() const {
  // Σ_i log Σ_k π_k(x_i) p(y_i | x_i, f_k, scale_k)
  // computed via logsumexp for numerical stability.
  if (num_data_ == 0) {
    return -std::numeric_limits<double>::infinity();
  }
  const label_t* labels = train_data_->metadata().label();
  const bool estimate_var = config_->mixture_estimate_variance;

  std::vector<double> log_norm(num_experts_, 0.0);
  std::vector<double> inv_scale(num_experts_, config_->mixture_e_step_alpha);
  if (estimate_var) {
    for (int k = 0; k < num_experts_; ++k) {
      const double s = std::max(expert_variance_[k], kMixtureEpsilon);
      if (e_step_loss_type_ == "l1") {
        log_norm[k]  = -std::log(2.0 * s);
        inv_scale[k] = 1.0 / s;
      } else if (e_step_loss_type_ == "l2") {
        log_norm[k]  = -0.5 * (std::log(2.0 * M_PI) + std::log(s));
        inv_scale[k] = 1.0 / (2.0 * s);
      } else {
        log_norm[k]  = 0.0;
        inv_scale[k] = 1.0 / s;
      }
    }
  }

  double total = 0.0;
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:total)
  for (data_size_t i = 0; i < num_data_; ++i) {
    double max_term = -std::numeric_limits<double>::infinity();
    std::vector<double> terms(num_experts_);
    for (int k = 0; k < num_experts_; ++k) {
      // Prior π_k(x) is bias-free, matching the E-step. ELBO is the model
      // log-likelihood; the routing-side bias is not part of the model.
      const double pi  = gate_proba_no_bias_[i * num_experts_ + k];
      const double f   = expert_pred_sm_[i * num_experts_ + k];
      const double y   = static_cast<double>(labels[i]);
      double loss;
      if (e_step_loss_type_ == "l1") {
        loss = std::fabs(y - f);
      } else if (e_step_loss_type_ == "quantile") {
        // Quantile has no proper density. We treat the asymmetric pinball
        // loss as a Laplace-style log-density exponent (matches the E-step
        // and UpdateExpertVariances). Without this branch the ELBO used
        // squared residuals while EStep used pinball — different "models"
        // in two diagnostics meant the logged number was uninterpretable.
        loss = ComputePointwiseLoss(y, f);
      } else {
        loss = (y - f) * (y - f);
      }
      const double t = std::log(pi + kMixtureEpsilon)
                     + log_norm[k] - inv_scale[k] * loss;
      terms[k] = t;
      if (t > max_term) max_term = t;
    }
    double sum_exp = 0.0;
    for (int k = 0; k < num_experts_; ++k) {
      sum_exp += std::exp(terms[k] - max_term);
    }
    total += max_term + std::log(sum_exp + kMixtureEpsilon);
  }
  return total;
}

void MixtureGBDT::UpdateExpertLoad() {
  // Compute current load per expert (mean responsibility)
  std::fill(expert_load_.begin(), expert_load_.end(), 0.0);

  for (data_size_t i = 0; i < num_data_; ++i) {
    for (int k = 0; k < num_experts_; ++k) {
      expert_load_[k] += responsibilities_[i * num_experts_ + k];
    }
  }

  for (int k = 0; k < num_experts_; ++k) {
    expert_load_[k] /= num_data_;  // Normalize to [0, 1], should sum to 1
  }

  // Log for debugging
  if (iter_ % 10 == 0 && config_->mixture_load_balance_alpha > 0.0) {
    std::string load_str = "";
    for (int k = 0; k < num_experts_; ++k) {
      load_str += std::to_string(expert_load_[k]).substr(0, 5) + " ";
    }
    Log::Debug("MixtureGBDT: Expert loads (aux) = [%s]", load_str.c_str());
  }
}

void MixtureGBDT::EStepExpertChoice() {
  // Step 1: Compute affinity scores
  ComputeAffinityScores();

  // Step 2: Each expert selects top-C samples
  SelectTopSamplesPerExpert();

  // Step 3: Convert selection to soft responsibilities
  ConvertSelectionToResponsibilities();
}

void MixtureGBDT::ComputeAffinityScores() {
  const label_t* labels = train_data_->metadata().label();
  const double alpha = config_->mixture_e_step_alpha;
  const std::string& score_type = config_->mixture_expert_choice_score;

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    for (int k = 0; k < num_experts_; ++k) {
      double score = 0.0;

      if (score_type == "gate" || score_type == "combined") {
        // Affinity = log π_k − α·loss is the same "score" form as the E-step:
        // bias does not enter the prior, only routing. Reading the bias-free
        // prior keeps Expert-Choice's affinity consistent with Token-Choice.
        double gate_prob = gate_proba_no_bias_[i * num_experts_ + k];
        score += std::log(gate_prob + kMixtureEpsilon);
      }

      if (score_type == "loss" || score_type == "combined") {
        double expert_p = expert_pred_sm_[i * num_experts_ + k];
        double loss = ComputePointwiseLoss(labels[i], expert_p);
        score -= alpha * loss;
      }

      // Note: expert_bias_ is already applied in Forward() via gate softmax.
      // Do NOT add it here to avoid double application.

      affinity_scores_[i * num_experts_ + k] = score;
    }
  }
}

void MixtureGBDT::SelectTopSamplesPerExpert() {
  std::fill(expert_selection_mask_.begin(),
            expert_selection_mask_.end(), 0);

  // Compute score statistics to set appropriate noise scale
  double score_std = 0.0;
  {
    double sum = 0.0, sum_sq = 0.0;
    size_t count = static_cast<size_t>(num_data_) * num_experts_;
    for (size_t i = 0; i < count; ++i) {
      sum += affinity_scores_[i];
      sum_sq += affinity_scores_[i] * affinity_scores_[i];
    }
    double mean = sum / count;
    double variance = sum_sq / count - mean * mean;
    score_std = std::sqrt(std::max(variance, 1e-10));
  }

  // Adaptive noise scale based on score variance
  // Early iterations: higher noise (when scores are similar)
  // Later iterations: lower noise (when scores are more distinct)
  const int warmup_iters = config_->mixture_warmup_iters;
  double noise_scale;
  if (iter_ < warmup_iters) {
    // During warmup: noise proportional to score std, with minimum
    // This ensures differentiation even when scores are nearly identical
    noise_scale = std::max(score_std * 0.5, 0.1);
  } else {
    // After warmup: small noise for tie-breaking only
    noise_scale = score_std * 0.01 + 1e-6;
  }

  for (int k = 0; k < num_experts_; ++k) {
    // Use expert-specific seed for noise (different each iteration and expert)
    std::mt19937 rng(config_->seed + k * 10000 + iter_ * 100);
    std::normal_distribution<double> noise_dist(0.0, noise_scale);

    // Collect (score + noise, index) pairs for this expert
    std::vector<std::pair<double, data_size_t>> scores_idx(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) {
      double score = affinity_scores_[i * num_experts_ + k];
      // Add noise to break ties and force differentiation
      score += noise_dist(rng);
      scores_idx[i] = {score, i};
    }

    // Partial sort for top-C (O(N) average)
    int C = std::min(expert_capacity_, static_cast<int>(num_data_));
    std::nth_element(
        scores_idx.begin(),
        scores_idx.begin() + C,
        scores_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Mark selected samples
    for (int c = 0; c < C; ++c) {
      data_size_t idx = scores_idx[c].second;
      expert_selection_mask_[idx * num_experts_ + k] = 1;
    }
  }
}

void MixtureGBDT::ConvertSelectionToResponsibilities() {
  const double boost = config_->mixture_expert_choice_boost;
  const bool hard_routing = config_->mixture_expert_choice_hard;
  const double min_r = hard_routing ? 0.0 : (1.0 / (num_experts_ * boost));

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    double sum = 0.0;
    int num_selected = 0;

    // Assign based on selection
    for (int k = 0; k < num_experts_; ++k) {
      size_t idx = i * num_experts_ + k;
      if (expert_selection_mask_[idx] == 1) {
        // Selected: high responsibility
        if (hard_routing) {
          // Hard routing: use affinity score directly (will be normalized)
          responsibilities_[idx] = std::exp(affinity_scores_[idx]);
        } else {
          // Soft routing: boost selected samples
          responsibilities_[idx] = std::exp(affinity_scores_[idx]) * boost;
        }
        num_selected++;
      } else {
        // Not selected: zero (hard) or minimum (soft)
        responsibilities_[idx] = min_r;
      }
      sum += responsibilities_[idx];
    }

    // Handle edge case: sample not selected by any expert (hard routing)
    if (hard_routing && num_selected == 0) {
      // Uniform distribution as fallback
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] = 1.0 / num_experts_;
      }
    } else {
      // Normalize (sum = 1). Epsilon floor is defensive: in practice sum > 0
      // because soft routing guarantees min_r > 0 for non-selected experts and
      // hard routing's num_selected == 0 path is handled above. Underflow is
      // only possible if every selected expert has affinity ≪ -700 (exp → 0)
      // — vanishingly unlikely given gate logits are bounded — but the cost
      // of floor()ing is zero and it forecloses one NaN-propagation route.
      const double inv_sum = 1.0 / std::max(sum, kMixtureEpsilon);
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] *= inv_sum;
      }
    }
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
      // Renormalize. Mathematically sum ≡ 1 here (convex combination of two
      // already-normalized rows), so the epsilon floor is purely defensive
      // against floating-point pathologies — but it matches the momentum
      // branch below, which has carried the same guard for a while.
      const double inv_sum = 1.0 / std::max(sum, kMixtureEpsilon);
      for (int k = 0; k < num_experts_; ++k) {
        responsibilities_[i * num_experts_ + k] *= inv_sum;
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
  // This is critical for regime-switching data where regimes may be genuinely
  // imbalanced (e.g., 70:30). Bidirectional balancing would fight against
  // the correct solution.
  //
  // min_usage = 1 / (balance_factor * K)
  // e.g., factor=10, K=2 -> min_usage = 5%, allows 95:5 imbalance

  const double min_usage = 1.0 / (config_->mixture_balance_factor * num_experts_);
  const double bias_update_rate = 0.1;
  // Decay rate for healthy experts. Bias is a *corrective* force — once an
  // expert is no longer underloaded, its bias should drift back to zero so
  // the gate's own signal can take over. Without decay, bias accumulated
  // monotonically across iterations and eventually dominated the gate
  // softmax, making the gate's learning effectively a no-op late in training.
  const double bias_decay_rate = 0.02;

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

  // Bidirectional update with decay:
  //   - underloaded (load < min_usage): push bias up to recover
  //   - healthy (load >= min_usage): exponentially decay bias toward 0
  // Natural regime imbalance (e.g. 70:30) still survives because as long as
  // both experts are above min_usage, neither bias is forced anywhere — they
  // simply decay back to whatever the gate's own logits naturally produce.
  for (int k = 0; k < num_experts_; ++k) {
    if (actual_load[k] < min_usage) {
      const double load_diff = min_usage - actual_load[k];
      expert_bias_[k] += bias_update_rate * load_diff;
    } else {
      expert_bias_[k] *= (1.0 - bias_decay_rate);
    }
  }

  // Log for debugging (only occasionally to avoid spam)
  if (iter_ % 10 == 0) {
    std::string load_str = "";
    std::string bias_str = "";
    for (int k = 0; k < num_experts_; ++k) {
      load_str += std::to_string(actual_load[k]).substr(0, 5) + " ";
      bias_str += std::to_string(expert_bias_[k]).substr(0, 6) + " ";
    }
    Log::Debug("MixtureGBDT: Expert loads = [%s], bias = [%s]",
               load_str.c_str(), bias_str.c_str());
  }
}

void MixtureGBDT::MStepExperts() {
  const label_t* labels = train_data_->metadata().label();

  // Each expert should optimize its OWN prediction toward the label
  // Gradient for expert k: r_ik * d_L(y_i, f_k(x_i)) / d_f_k(x_i)
  // This ensures experts specialize on different parts of the data

  // Expert Dropout: randomly skip some experts to prevent collapse
  // Supports curriculum scheduling: low dropout early, higher late
  double dropout_rate = config_->mixture_expert_dropout_rate;
  if (config_->mixture_dropout_schedule != "constant" && config_->num_iterations > 0) {
    const int moe_iter = use_progressive_ ? (iter_ - seed_iterations_) : iter_;
    const int total_moe = use_progressive_
        ? (config_->num_iterations - seed_iterations_)
        : config_->num_iterations;
    const double progress = std::min(1.0, std::max(0.0,
        static_cast<double>(moe_iter) / std::max(1, total_moe)));
    const double lo = config_->mixture_dropout_rate_min;
    const double hi = config_->mixture_dropout_rate_max;
    if (config_->mixture_dropout_schedule == "linear") {
      dropout_rate = lo + (hi - lo) * progress;
    } else if (config_->mixture_dropout_schedule == "cosine") {
      dropout_rate = lo + (hi - lo) * 0.5 * (1.0 - std::cos(progress * M_PI));
    }
  }

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

  // Precompute argmax assignment for hard M-step
  const bool hard_m_step = config_->mixture_hard_m_step;
  const double diversity_lambda = config_->mixture_diversity_lambda;
  std::vector<int> best_expert(num_data_, 0);

  // Per-expert sample index lists for sparse activation. Stored as a member
  // (expert_sample_indices_) so the buffer outlives the MStepExperts call —
  // the pointer passed to SetBaggingData is held by data_partition_ and may
  // be re-read on the next BeforeTrain() if no fresh SetBaggingData is made.
  if (static_cast<int>(expert_sample_indices_.size()) != num_experts_) {
    expert_sample_indices_.assign(num_experts_, std::vector<data_size_t>());
  }
  for (int k = 0; k < num_experts_; ++k) {
    expert_sample_indices_[k].clear();
  }

  if (hard_m_step) {
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      double max_r = -1.0;
      for (int k = 0; k < num_experts_; ++k) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        if (r_ik > max_r) {
          max_r = r_ik;
          best_expert[i] = k;
        }
      }
    }

    // Build per-expert sample index lists for sparse histogram construction
    for (int k = 0; k < num_experts_; ++k) {
      expert_sample_indices_[k].reserve(num_data_ / num_experts_);
    }
    for (data_size_t i = 0; i < num_data_; ++i) {
      expert_sample_indices_[best_expert[i]].push_back(i);
    }
  }

  // Phase 1: Pre-compute gradients for all experts (using full OMP parallelism)
  std::vector<std::vector<score_t>> all_grads(num_experts_);
  std::vector<std::vector<score_t>> all_hess(num_experts_);

  for (int k = 0; k < num_experts_; ++k) {
    if (!expert_active[k]) {
      // Dropped experts get zero gradients (trivial no-split tree)
      all_grads[k].assign(num_data_, 0.0);
      all_hess[k].assign(num_data_, kMixtureEpsilon);
      continue;
    }

    all_grads[k].resize(num_data_);
    all_hess[k].resize(num_data_);

    const double* expert_k_pred = expert_pred_.data() + k * num_data_;

    if (objective_function_ != nullptr) {
      std::vector<double> expert_k_pred_vec(expert_k_pred, expert_k_pred + num_data_);
      std::vector<score_t> temp_grad(num_data_);
      std::vector<score_t> temp_hess(num_data_);
      objective_function_->GetGradients(expert_k_pred_vec.data(), temp_grad.data(), temp_hess.data());

      // Gradient weighting is always soft (r_ik). The hard_m_step flag now
      // only restricts which samples each expert sees via SetBaggingData
      // (sparse subset of argmax winners) — see the bagging branch above.
      // Earlier behavior zeroed gradients for non-winners outright, which
      // produced expert collapse whenever bagging fell back to the full
      // dataset (assigned < min_safe): losers got no gradient signal at all
      // for those samples, so the gate could not learn to route to them.
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = responsibilities_[i * num_experts_ + k];
        all_grads[k][i] = static_cast<score_t>(r_ik * temp_grad[i]);
        const double hess_val = r_ik * static_cast<double>(temp_hess[i]);
        all_hess[k][i] = static_cast<score_t>(std::max(hess_val, kMixtureEpsilon));

        if (diversity_lambda > 0.0) {
          // Diversity regularizer — encourages f_k to differ from f_j on
          // samples j owns (high r_ij). Earlier code added
          //     +λ Σ_{j≠k} r_ij (f_k − f_j)
          // to the gradient — that is the gradient of *aligning* f_k with
          // the other experts, not diversifying them. Verified empirically:
          // at λ=0.05 with K=3 the old sign drove pairwise expert distance
          // down to 22% of the λ=0 baseline.
          //
          // Sign-flipping alone is unstable (the natural diversity reward
          // R_k = -½ λ Σ r_ij (f_k − f_j)² is unbounded below; predictions
          // run off to ±∞ at any non-trivial λ — λ=0.001 was empirically
          // shown to inflate peak |pred| 25× in 30 iters). Huber-style
          // saturation: clip (f_k − f_j) to ±δ before summing so the per-
          // pair contribution is bounded by ±λ·δ·r_ij. Inside the clip
          // region the reward is quadratic; outside it is linear. Hessian
          // damping +λ·Σ r_ij keeps the leaf-value Newton step
          // well-conditioned (the un-saturated true Hessian of a diversity
          // reward is negative, which destabilizes Newton).
          constexpr double kDiversityClip = 1.0;
          double div_grad = 0.0;
          double div_hess = 0.0;
          for (int j = 0; j < num_experts_; ++j) {
            if (j == k) continue;
            const double r_ij = responsibilities_[i * num_experts_ + j];
            const double f_k = expert_k_pred[i];
            const double f_j = expert_pred_sm_[i * num_experts_ + j];
            const double diff = f_k - f_j;
            const double clipped = std::max(-kDiversityClip,
                                            std::min(kDiversityClip, diff));
            div_grad += r_ij * clipped;
            div_hess += r_ij;
          }
          const double inv_pairs = 1.0 / (num_experts_ - 1);
          all_grads[k][i] -= static_cast<score_t>(diversity_lambda * div_grad * inv_pairs);
          all_hess[k][i] += static_cast<score_t>(diversity_lambda * div_hess * inv_pairs);
        }
      }
    } else {
      // L2 fallback path: same soft-gradient policy as above.
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        double diff = expert_k_pred[i] - labels[i];
        double r_ik = responsibilities_[i * num_experts_ + k];
        all_grads[k][i] = static_cast<score_t>(r_ik * 2.0 * diff);
        all_hess[k][i] = static_cast<score_t>(
            std::max(r_ik * 2.0, kMixtureEpsilon));

        if (diversity_lambda > 0.0) {
          // Same Huber-saturated diversity regularizer as the
          // objective_function_ branch above. See that branch for the
          // sign / clip / Hessian damping rationale.
          constexpr double kDiversityClip = 1.0;
          double div_grad = 0.0;
          double div_hess = 0.0;
          for (int j = 0; j < num_experts_; ++j) {
            if (j == k) continue;
            const double r_ij = responsibilities_[i * num_experts_ + j];
            const double f_k = expert_k_pred[i];
            const double f_j = expert_pred_sm_[i * num_experts_ + j];
            const double dd = f_k - f_j;
            const double clipped = std::max(-kDiversityClip,
                                            std::min(kDiversityClip, dd));
            div_grad += r_ij * clipped;
            div_hess += r_ij;
          }
          const double inv_pairs = 1.0 / (num_experts_ - 1);
          all_grads[k][i] -= static_cast<score_t>(diversity_lambda * div_grad * inv_pairs);
          all_hess[k][i] += static_cast<score_t>(diversity_lambda * div_hess * inv_pairs);
        }
      }
    }
  }

  // Adaptive per-expert learning rate: scale gradients based on loss trend
  if (config_->mixture_adaptive_lr) {
    const label_t* lr_labels = train_data_->metadata().label();
    const int window = config_->mixture_adaptive_lr_window;
    const double max_scale = config_->mixture_adaptive_lr_max;
    const double min_scale = 1.0 / max_scale;

    // Compute current mean loss per expert (over assigned samples)
    for (int k = 0; k < num_experts_; ++k) {
      if (!expert_active[k]) continue;
      const double* expert_k_pred = expert_pred_.data() + k * num_data_;
      double total_loss = 0.0;
      int count = 0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        double r_ik = hard_m_step
            ? (best_expert[i] == k ? 1.0 : 0.0)
            : responsibilities_[i * num_experts_ + k];
        if (r_ik > 0.1) {  // Only count significantly assigned samples
          total_loss += ComputePointwiseLoss(lr_labels[i], expert_k_pred[i]);
          ++count;
        }
      }
      expert_loss_history_[k][loss_history_pos_ % window] =
          count > 0 ? total_loss / count : 0.0;
    }
    ++loss_history_pos_;

    // Compute LR scale from loss trend (after filling at least half the window)
    if (loss_history_pos_ >= window / 2) {
      for (int k = 0; k < num_experts_; ++k) {
        int filled = std::min(loss_history_pos_, window);
        // Compare first half mean vs second half mean
        double first_half = 0.0, second_half = 0.0;
        int half = filled / 2;
        for (int t = 0; t < half; ++t) {
          int idx = (loss_history_pos_ - filled + t) % window;
          first_half += expert_loss_history_[k][idx];
        }
        for (int t = half; t < filled; ++t) {
          int idx = (loss_history_pos_ - filled + t) % window;
          second_half += expert_loss_history_[k][idx];
        }
        first_half /= std::max(1, half);
        second_half /= std::max(1, filled - half);

        // If loss is decreasing (improving) → lower LR (fine-tune)
        // If loss is increasing or stagnating → higher LR (escape)
        double ratio = (first_half > 1e-10) ? second_half / first_half : 1.0;
        // ratio < 1 → improving, ratio > 1 → worsening
        // Map ratio to LR scale: improving → min_scale, worsening → max_scale
        double scale = std::max(min_scale, std::min(max_scale, ratio));
        expert_lr_scale_[k] = scale;
      }
    }

    // Apply LR scale to gradients
    for (int k = 0; k < num_experts_; ++k) {
      if (!expert_active[k]) continue;
      double scale = expert_lr_scale_[k];
      if (std::abs(scale - 1.0) > 1e-6) {
        score_t s = static_cast<score_t>(scale);
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i) {
          all_grads[k][i] *= s;
        }
      }
    }

    // Log occasionally
    if (iter_ % 20 == 0) {
      std::string lr_str;
      for (int k = 0; k < num_experts_; ++k) {
        lr_str += std::to_string(expert_lr_scale_[k]).substr(0, 5) + " ";
      }
      Log::Debug("MixtureGBDT: Adaptive LR scales = [%s]", lr_str.c_str());
    }
  }

  // Lazily build the [0..num_data_-1] identity index buffer used as the
  // "no sparse restriction" fallback for SetBaggingData. Holding it on the
  // class keeps the pointer stable across iterations (data_partition_ caches
  // the pointer until SetBaggingData is called again).
  if (static_cast<data_size_t>(all_data_indices_.size()) != num_data_) {
    all_data_indices_.resize(num_data_);
    std::iota(all_data_indices_.begin(), all_data_indices_.end(),
              static_cast<data_size_t>(0));
  }

  // Phase 2: Train experts sequentially.
  //
  // Earlier versions trained experts concurrently via std::thread (#9), but
  // running OpenMP parallel regions from multiple non-OMP host threads
  // simultaneously crashes inside libgomp on certain Optuna parameter
  // combinations (see issue #16 — segfault in LeafSplits::Init's OMP fn,
  // plus the spurious "num_threads changed during training" warnings caused
  // by mutating LGBM_DEFAULT_NUM_THREADS, a process global, from the host
  // thread). Each expert's tree learner uses its own OpenMP team internally,
  // so sequential training keeps the inner parallelism while avoiding the
  // unsafe nested host-thread + OMP interaction.
  for (int k = 0; k < num_experts_; ++k) {
    // Always call SetBaggingData with a member-owned buffer. data_partition_
    // stores the raw pointer and re-reads it on every BeforeTrain(). Passing
    // a stack pointer (the original #10 implementation) left a dangling
    // reference once MStepExperts returned, leading to the segfault in #16.
    //
    // Sparse activation: restrict to assigned samples only when there's
    // enough data — SerialTreeLearner can otherwise pick a "best" split with
    // one side empty, tripping CHECK_GT(*_count, 0) at
    // serial_tree_learner.cpp:859/869. Falling back to the full dataset
    // (with zero gradients on non-assigned rows) is the pre-#10 behavior
    // and is safe for tiny experts.
    const int min_data = expert_configs_[k]->min_data_in_leaf;
    const int min_safe = std::max(2 * min_data, 16);
    const data_size_t assigned =
        static_cast<data_size_t>(expert_sample_indices_[k].size());
    if (hard_m_step && assigned >= static_cast<data_size_t>(min_safe)) {
      experts_[k]->SetBaggingData(expert_sample_indices_[k].data(), assigned);
    } else {
      experts_[k]->SetBaggingData(all_data_indices_.data(), num_data_);
    }
    try {
      experts_[k]->TrainOneIter(all_grads[k].data(), all_hess[k].data());
    } catch (const std::exception& e) {
      Log::Warning("MixtureGBDT: expert %d skipped iter %d (%s)",
                   k, iter_, e.what());
    } catch (...) {
      Log::Warning("MixtureGBDT: expert %d skipped iter %d (unknown error)",
                   k, iter_);
    }
  }
}

void MixtureGBDT::MStepGate() {
  // Soft cross-entropy against the full responsibility distribution r_ik.
  //
  // The gate produces raw logits z_ik. At routing time those are combined
  // with `expert_bias_` (load-balancing nudge) and `gate_temperature_`
  // (annealing) before softmax: routing_prob = softmax((z + b) / T).
  //
  // Design choices in the gradient below:
  //
  //  (a) Gradient is computed against `softmax(z / T)` *without* bias. We
  //      want the gate's own logits to fit r directly; if bias enters the
  //      training target, the gate would spend capacity each iter undoing
  //      the bias the load-balancer just added (DeepSeek "Auxiliary-Loss-Free
  //      Load Balancing" applies bias only to the routing decision, never
  //      to the gate's training target).
  //
  //  (b) Chain rule through the temperature: with logit `u = z / T` and
  //      `p = softmax(u)`, the cross-entropy loss against target r has
  //          dL/dz = (1/T)(p − r),    d²L/dz² = (1/T²) p(1 − p).
  //      Earlier code used `p − r` and `p(1 − p)` directly, mis-scaling the
  //      Newton step by T at non-unit temperatures.
  //
  //  (c) Friedman's K/(K-1) factor on the Hessian (matches standard
  //      LightGBM `MulticlassSoftmax::GetGradients` in
  //      multiclass_objective.hpp). Corrects for the (K-1) effective degrees
  //      of freedom in K softmax logits — without it, Newton leaf values are
  //      a factor (K-1)/K of standard (e.g. 2/3 at K=3, 9/10 at K=10).
  //
  //  (d) Grad/hess are recomputed inside the `gate_iters_per_round` loop.
  //      Each `gate_->TrainOneIter` adds K trees (one per class), so z and
  //      hence p change between iterations. Reusing the iter-1 grad/hess on
  //      iter 2+ would degrade Newton's method to constant-gradient
  //      subgradient descent. Default `gate_iters_per_round=1` made this a
  //      latent issue; the recompute is cheap so we always do it.
  //
  // Earlier versions of this code collapsed r_i to a one-hot via argmax
  // before computing CE, which discarded the soft routing signal: that fix
  // landed in PR #23 (Jordan-Jacobs soft EM) — kept here.
  std::vector<score_t> gate_grad(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<score_t> gate_hess(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<double> gate_raw_no_bias(
      static_cast<size_t>(num_data_) * num_experts_);

  // Dirichlet-shrinkage regularizer (kept under the legacy
  // `mixture_gate_entropy_lambda` parameter name for back-compat — see the
  // audit note in the header for why the gradient `λ(p − 1/K)` corresponds
  // to a Dirichlet shrinkage toward uniform, not the entropy gradient
  // d(−H)/dz which would vanish near simplex corners and so be a weak
  // anti-collapse signal). Same /T chain rule as the base gradient.
  const double dirichlet_lambda = config_->mixture_gate_entropy_lambda;
  const double uniform_prob = 1.0 / num_experts_;
  const double T = std::max(gate_temperature_, kMixtureEpsilon);
  const double inv_T = 1.0 / T;
  const double inv_T2 = inv_T * inv_T;
  // Friedman's K/(K-1) factor — see (c) above. Guarded for K=1 to avoid
  // division by zero, though K=1 makes MoE itself meaningless.
  const double friedman_factor = (num_experts_ > 1)
      ? static_cast<double>(num_experts_) / (num_experts_ - 1.0)
      : 1.0;

  for (int g = 0; g < config_->mixture_gate_iters_per_round; ++g) {
    // (d) Refresh raw logits each iter — previous TrainOneIter mutated z.
    {
      int64_t out_len;
      gate_->GetPredictAt(0, gate_raw_no_bias.data(), &out_len);
    }

    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      // Per-sample bias-free softmax. Stack buffer for K ≤ 64 to avoid the
      // per-iter heap thrash of `std::vector<double> p(num_experts_)`.
      double scores_buf[64];
      double p_buf[64];
      double* scores = (num_experts_ <= 64) ? scores_buf
                                            : new double[num_experts_];
      double* p = (num_experts_ <= 64) ? p_buf : new double[num_experts_];

      for (int k = 0; k < num_experts_; ++k) {
        scores[k] = gate_raw_no_bias[k * num_data_ + i] * inv_T;
      }
      Softmax(scores, num_experts_, p);

      for (int k = 0; k < num_experts_; ++k) {
        size_t idx = i + k * num_data_;  // Gate uses class-major order
        const double r = responsibilities_[i * num_experts_ + k];

        // Chain-rule-correct gradient on z (logit-space). Both base CE and
        // Dirichlet shrinkage are scaled by 1/T.
        const double base_grad = (p[k] - r) * inv_T;
        const double reg_grad =
            dirichlet_lambda * (p[k] - uniform_prob) * inv_T;

        gate_grad[idx] = static_cast<score_t>(base_grad + reg_grad);

        // Diagonal Hessian: softmax CE on the K-redundant softmax has
        //   K/(K-1) · p(1-p) / T²
        // (the Friedman factor from the standard multiclass objective). The
        // Dirichlet term's exact diagonal Hessian is λ p(1-p)/T² which would
        // vanish at corners and explode the Newton step there; we replace it
        // with a constant λ/T² damping. Both are floored at kMixtureEpsilon
        // for numerical safety.
        gate_hess[idx] = static_cast<score_t>(
            std::max((friedman_factor * p[k] * (1.0 - p[k]) + dirichlet_lambda)
                         * inv_T2,
                     kMixtureEpsilon));
      }

      if (num_experts_ > 64) {
        delete[] scores;
        delete[] p;
      }
    }

    gate_->TrainOneIter(gate_grad.data(), gate_hess.data());
  }

  // Log Dirichlet-shrinkage effect (occasionally). Reads gate_proba_ which
  // includes bias — that's intentional, this metric describes the actual
  // routing distribution's entropy, not the bias-free gate output.
  if (dirichlet_lambda > 0.0 && iter_ % 10 == 0) {
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
    Log::Debug("MixtureGBDT: Gate Dirichlet-shrinkage active (lambda=%.3f), "
               "avg normalized routing entropy=%.3f",
               dirichlet_lambda, normalized_entropy);
  }
}

void MixtureGBDT::MStepGateLeafReuse() {
  // Derive gate probabilities from expert tree leaf statistics.
  // Uses bin data directly — no raw features or extra memory needed.
  //
  // Algorithm:
  // 1. Traverse expert 0's latest tree using bin values from Dataset
  //    to get each sample's leaf index
  // 2. For each leaf, aggregate responsibility distribution across experts
  // 3. Set gate_proba_ from per-leaf routing statistics
  // 4. Periodically retrain gate GBDT for inference on new data

  const Tree* routing_tree = experts_[0]->GetLatestTree();
  if (routing_tree == nullptr || routing_tree->num_leaves() <= 1) {
    MStepGate();
    return;
  }

  const int num_leaves = routing_tree->num_leaves();
  const int num_internal = num_leaves - 1;  // number of split nodes
  const int num_features = train_data_->num_features();

  // Build a flat lookup table BinIterator* indexed by inner_feat. Replaces a
  // per-thread std::unordered_map lookup that previously fired on every node
  // traversal — at 300 trials × 5 CV × 100 rounds × 2000 samples × tree depth
  // the hash lookup dominated runtime (issue #16 follow-up).
  //
  // BinIterator::RawGet is a stateless read of bin_data_, so a single iterator
  // per feature is shared safely across OMP threads. Iterators are also held
  // as a class member (leaf_reuse_iters_) so they are allocated once per
  // tree-shape change instead of per call.
  if (static_cast<int>(leaf_reuse_iters_.size()) != num_features) {
    for (BinIterator* p : leaf_reuse_iters_) {
      delete p;
    }
    leaf_reuse_iters_.assign(num_features, nullptr);
    leaf_reuse_iter_features_.clear();
  }

  // Track which features have iterators allocated, so only those are torn down
  // when the tree shape changes. We rebuild on each call — cheap because the
  // set of split features is small (≤ num_internal).
  for (int inner_feat : leaf_reuse_iter_features_) {
    delete leaf_reuse_iters_[inner_feat];
    leaf_reuse_iters_[inner_feat] = nullptr;
  }
  leaf_reuse_iter_features_.clear();
  for (int node = 0; node < num_internal; ++node) {
    const int inner_feat = routing_tree->split_feature_inner(node);
    if (leaf_reuse_iters_[inner_feat] == nullptr) {
      leaf_reuse_iters_[inner_feat] = train_data_->FeatureIterator(inner_feat);
      leaf_reuse_iter_features_.push_back(inner_feat);
    }
  }

  // Step 1: Get leaf index for each sample via bin-based tree traversal.
  // Pre-cache split metadata in flat arrays so the inner loop touches only
  // raw int reads (no virtual calls per node, no unordered_map lookups).
  std::vector<BinIterator*> node_iter(num_internal);
  std::vector<uint32_t> node_threshold(num_internal);
  std::vector<int> node_left(num_internal);
  std::vector<int> node_right(num_internal);
  for (int node = 0; node < num_internal; ++node) {
    node_iter[node] = leaf_reuse_iters_[routing_tree->split_feature_inner(node)];
    node_threshold[node] = routing_tree->threshold_in_bin(node);
    node_left[node] = routing_tree->left_child(node);
    node_right[node] = routing_tree->right_child(node);
  }

  if (static_cast<data_size_t>(sample_leaf_buf_.size()) != num_data_) {
    sample_leaf_buf_.resize(num_data_);
  }
  int* sample_leaf = sample_leaf_buf_.data();

  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int node = 0;
    while (node >= 0) {
      const uint32_t bin_val = node_iter[node]->RawGet(i);
      node = (bin_val <= node_threshold[node])
                 ? node_left[node]
                 : node_right[node];
    }
    sample_leaf[i] = ~node;  // leaf index
  }

  // Step 2: Aggregate responsibility distribution per leaf using a flat
  // [num_leaves * num_experts_] buffer (member-owned to avoid per-call allocs).
  const size_t leaf_buf_sz = static_cast<size_t>(num_leaves) * num_experts_;
  if (leaf_expert_sum_buf_.size() < leaf_buf_sz) {
    leaf_expert_sum_buf_.resize(leaf_buf_sz);
  }
  if (static_cast<int>(leaf_count_buf_.size()) < num_leaves) {
    leaf_count_buf_.resize(num_leaves);
  }
  std::fill(leaf_expert_sum_buf_.begin(),
            leaf_expert_sum_buf_.begin() + leaf_buf_sz, 0.0);
  std::fill(leaf_count_buf_.begin(),
            leaf_count_buf_.begin() + num_leaves, 0);

  for (data_size_t i = 0; i < num_data_; ++i) {
    const int leaf = sample_leaf[i];
    if (leaf >= 0 && leaf < num_leaves) {
      ++leaf_count_buf_[leaf];
      const size_t leaf_off = static_cast<size_t>(leaf) * num_experts_;
      const size_t resp_off = static_cast<size_t>(i) * num_experts_;
      for (int k = 0; k < num_experts_; ++k) {
        leaf_expert_sum_buf_[leaf_off + k] += responsibilities_[resp_off + k];
      }
    }
  }

  // Normalize to per-leaf routing probabilities (in-place into the same buf).
  const double uniform = 1.0 / num_experts_;
  for (int l = 0; l < num_leaves; ++l) {
    const size_t off = static_cast<size_t>(l) * num_experts_;
    double sum = 0.0;
    if (leaf_count_buf_[l] > 0) {
      for (int k = 0; k < num_experts_; ++k) sum += leaf_expert_sum_buf_[off + k];
    }
    if (sum > 0.0) {
      const double inv = 1.0 / sum;
      for (int k = 0; k < num_experts_; ++k) {
        leaf_expert_sum_buf_[off + k] *= inv;
      }
    } else {
      for (int k = 0; k < num_experts_; ++k) {
        leaf_expert_sum_buf_[off + k] = uniform;
      }
    }
  }

  // Step 3: Set gate_proba_ from leaf routing statistics
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_data_; ++i) {
    int leaf = sample_leaf[i];
    if (leaf >= 0 && leaf < num_leaves) {
      const size_t leaf_off = static_cast<size_t>(leaf) * num_experts_;
      for (int k = 0; k < num_experts_; ++k) {
        gate_proba_[i * num_experts_ + k] = leaf_expert_sum_buf_[leaf_off + k];
      }
    }
  }

  // Step 4: Train the gate GBDT every iteration via soft CE against the
  // leaf-aggregated routing distribution. Earlier behavior trained the gate
  // only every mixture_gate_retrain_interval iterations against argmax
  // pseudo-labels, leaving the GBDT trees decorative: Forward read
  // gate_proba_ from leaf statistics during training, so the GBDT
  // contributed nothing to in-domain routing and was sparsely fit for
  // out-of-domain inference (PredictRegimeProba on unseen data).
  //
  // We compute the gate GBDT's *current* softmax predictions q_ik directly
  // (NOT gate_proba_, which was just overwritten with the leaf-aggregated
  // targets in Step 3). The CE gradient is then q_ik - target_ik, which
  // matches the same soft-EM gradient used by MStepGate in gbdt mode.
  std::vector<double> gate_raw_lr(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<score_t> gate_grad_lr(static_cast<size_t>(num_data_) * num_experts_);
  std::vector<score_t> gate_hess_lr(static_cast<size_t>(num_data_) * num_experts_);

  // Gate audit fixes mirroring MStepGate (gbdt path):
  //  (a) train against bias-free softmax — bias is for routing, not for the
  //      gate's training target (DeepSeek loss-free LB);
  //  (b) chain-rule scale gradient by 1/T and Hessian by 1/T²;
  //  (c) Friedman K/(K-1) factor on Hessian (matches standard multiclass);
  //  (d) refresh grad/hess inside the iter loop — see MStepGate for details.
  const double dirichlet_lambda_lr = config_->mixture_gate_entropy_lambda;
  const double uniform_prob_lr = 1.0 / num_experts_;
  const double T_lr = std::max(gate_temperature_, kMixtureEpsilon);
  const double inv_T_lr = 1.0 / T_lr;
  const double inv_T2_lr = inv_T_lr * inv_T_lr;
  const double friedman_factor_lr = (num_experts_ > 1)
      ? static_cast<double>(num_experts_) / (num_experts_ - 1.0)
      : 1.0;

  for (int g = 0; g < config_->mixture_gate_iters_per_round; ++g) {
    {
      int64_t out_len;
      gate_->GetPredictAt(0, gate_raw_lr.data(), &out_len);
    }

    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      // Bias-free softmax q_k = softmax(z_k / T) for the gradient target.
      std::vector<double> scores(num_experts_);
      for (int k = 0; k < num_experts_; ++k) {
        scores[k] = gate_raw_lr[k * num_data_ + i] * inv_T_lr;
      }
      std::vector<double> q(num_experts_);
      Softmax(scores.data(), num_experts_, q.data());

      // Resolve target distribution from the leaf assignment.
      const int leaf = sample_leaf[i];
      const double* target;
      std::vector<double> uniform_buf;
      if (leaf >= 0 && leaf < num_leaves) {
        target = leaf_expert_sum_buf_.data() + static_cast<size_t>(leaf) * num_experts_;
      } else {
        uniform_buf.assign(num_experts_, uniform_prob_lr);
        target = uniform_buf.data();
      }

      for (int k = 0; k < num_experts_; ++k) {
        const size_t idx = i + static_cast<size_t>(k) * num_data_;
        const double base_grad = (q[k] - target[k]) * inv_T_lr;
        const double reg_grad =
            dirichlet_lambda_lr * (q[k] - uniform_prob_lr) * inv_T_lr;
        gate_grad_lr[idx] = static_cast<score_t>(base_grad + reg_grad);
        gate_hess_lr[idx] = static_cast<score_t>(
            std::max((friedman_factor_lr * q[k] * (1.0 - q[k])
                          + dirichlet_lambda_lr) * inv_T2_lr,
                     kMixtureEpsilon));
      }
    }

    gate_->TrainOneIter(gate_grad_lr.data(), gate_hess_lr.data());
  }
}

bool MixtureGBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
  // MixtureGBDT ignores external gradients/hessians
  // (custom objective is handled internally via responsibility weighting)
  (void)gradients;
  (void)hessians;

  // === SEED PHASE (progressive mode) ===
  if (use_progressive_ && !seed_phase_complete_) {
    if (iter_ < seed_iterations_) {
      // Train seed expert on all data
      int64_t out_len;
      seed_expert_->GetPredictAt(0, yhat_.data(), &out_len);

      // Compute objective gradients on seed predictions
      if (objective_function_ != nullptr) {
        objective_function_->GetGradients(yhat_.data(), gradients_.data(), hessians_.data());
      } else {
        // Default MSE
        const label_t* labels = train_data_->metadata().label();
        for (data_size_t i = 0; i < num_data_; ++i) {
          double diff = yhat_[i] - labels[i];
          gradients_[i] = static_cast<score_t>(2.0 * diff);
          hessians_[i] = static_cast<score_t>(2.0);
        }
      }

      // Train seed expert
      seed_expert_->TrainOneIter(gradients_.data(), hessians_.data());

      // Update training predictions from seed
      seed_expert_->GetPredictAt(0, yhat_.data(), &out_len);

      // Update validation predictions from seed (needed for early stopping / metrics)
      for (size_t v = 0; v < valid_datas_.size(); ++v) {
        int64_t vlen;
        seed_expert_->GetPredictAt(static_cast<int>(v) + 1,
                                    yhat_valid_[v].data(), &vlen);
      }

      ++iter_;

      if (iter_ % 10 == 0 || iter_ == seed_iterations_) {
        Log::Info("MixtureGBDT: Seed phase iteration %d/%d",
                  iter_, seed_iterations_);
      }

      return false;
    } else {
      // Seed phase complete → spawn K experts
      SpawnExpertsFromSeed();
      seed_phase_complete_ = true;

      // Initialize responsibilities for the new experts
      InitResponsibilities();

      Log::Info("MixtureGBDT: Seed phase complete, entering MoE phase at iteration %d",
                iter_);
    }
  }

  // === Update gate temperature ===
  {
    int moe_iter = use_progressive_ ? (iter_ - seed_iterations_) : iter_;
    int total_moe_iters = use_progressive_
        ? (config_->num_iterations - seed_iterations_)
        : config_->num_iterations;
    gate_temperature_ = ComputeTemperature(moe_iter, total_moe_iters);

    if (moe_iter % 10 == 0 &&
        config_->mixture_gate_temperature_init != config_->mixture_gate_temperature_final) {
      Log::Info("MixtureGBDT: Gate temperature = %.4f (moe_iter=%d/%d)",
                gate_temperature_, moe_iter, total_moe_iters);
    }
  }

  // Forward pass - compute expert predictions and gate probabilities
  Forward();

  // E-step: update responsibilities
  // Skip E-step for first few iterations to allow experts to differentiate
  // with the initial responsibilities. Without this, all experts start with
  // prediction=0, causing EStep to compute uniform responsibilities and
  // all experts train identically.
  const int warmup_iters = config_->mixture_warmup_iters;
  const int moe_iter = use_progressive_ ? (iter_ - seed_iterations_) : iter_;

  const bool past_warmup = (moe_iter >= warmup_iters);

  if (past_warmup) {
    if (use_expert_choice_) {
      EStepExpertChoice();
    } else {
      EStep();
      UpdateExpertLoad();
    }

    // Apply time-series smoothing if enabled
    SmoothResponsibilities();

    // Update expert bias for loss-free load balancing
    UpdateExpertBias();

    // M-step for the per-expert noise scale σ_k² (or Laplace b_k). This
    // closes the EM loop on the parameter that the legacy code held fixed
    // via the temperature hyperparameter `mixture_e_step_alpha`. With the
    // scale estimated from the data, the responsibility softmax becomes the
    // proper Bayesian posterior of the mixture model rather than a hand-
    // tuned reweighting of expert losses.
    UpdateExpertVariances();
  }

  // M-step: update experts
  MStepExperts();

  // M-step: update gate (every iteration, including warmup).
  //
  // Earlier this skipped the gate during warmup on the theory that fitting
  // a frozen target (the EM-untouched r_init) would build trees that later
  // need "unlearning" once the E-step starts mutating r. In practice that
  // tradeoff was exactly backwards: r_init carries real structure (quantile
  // / kmeans / gmm partitioning of the data) and the gate sitting at uniform
  // softmax until iter == warmup_iters meant the first post-warmup E-step
  // used a *flat* prior π(x), throwing away the regime structure the init
  // had supplied. By the end of warmup_iters with this fix, the gate's
  // logits are ≈ log(r_init), so the first real E-step uses a meaningful
  // prior and the experts (already differentiated by warmup-time r-weighted
  // gradients) get a coherent routing signal. The "unlearning" cost is
  // bounded: trees built against r_init are still mostly correct under
  // small post-warmup updates because EM moves r incrementally from r_init.
  // (gate_type == "none" still skips entirely; it never trains a GBDT gate.)
  if (config_->mixture_gate_type == "gbdt") {
    MStepGate();
  } else if (config_->mixture_gate_type == "leaf_reuse") {
    MStepGateLeafReuse();
  }

  // ELBO / marginal log-likelihood diagnostic. EM with an exact M-step is
  // monotone non-decreasing in this quantity; here the M-step is
  // approximate (each expert / the gate adds one tree per iter), so
  // monotonicity is not guaranteed but should hold "most of the time".
  // A persistent decrease across many iters indicates the EM machinery is
  // not actually fitting the mixture — log it loudly so we notice.
  if (config_->mixture_estimate_variance &&
      moe_iter >= warmup_iters &&
      (iter_ % 10 == 0 || iter_ < 5)) {
    const double ll = ComputeMarginalLogLikelihood();
    Log::Info("MixtureGBDT: iter=%d  marginal_log_lik=%.6f  (per_sample=%.6f)",
              iter_, ll,
              num_data_ > 0 ? ll / num_data_ : 0.0);

    // Monotonicity check. Approximate M-step → small dips are normal, but a
    // drop bigger than 5% of |prev| is symptomatic of misalignment between
    // E-step and M-step — historically this has been a sign of:
    //   • bias-side regularizer fighting gate (pre PR #25)
    //   • diversity sign flip (pre PR #26)
    //   • dimension mismatch in scale estimation (pre this commit's quantile fix)
    //   • aggressive expert dropout / adaptive_lr decoupling experts from EM
    // The user can mute by quietening Log::Warning if expected (e.g. when
    // deliberately running with high dropout for ablation).
    const bool prev_finite = std::isfinite(prev_marginal_log_lik_) &&
                             prev_marginal_log_lik_ > -1e299;
    if (prev_finite && std::isfinite(ll)) {
      const double drop = prev_marginal_log_lik_ - ll;
      const double rel_scale = std::max(std::fabs(prev_marginal_log_lik_), 1.0);
      if (drop > 0.05 * rel_scale) {
        Log::Warning(
            "MixtureGBDT: marginal_log_lik dropped %.6f → %.6f "
            "(Δ=%.6f, %.1f%% of |prev|). Approximate M-step allows small "
            "dips; persistent / large drops mean E-step and M-step are "
            "optimizing inconsistent objectives. Suspect: high expert "
            "dropout, mixture_adaptive_lr, aggressive temperature "
            "annealing, or a recent code change to gradient/Hessian.",
            prev_marginal_log_lik_, ll, -drop,
            100.0 * drop / rel_scale);
      }
    }
    prev_marginal_log_lik_ = ll;
  }

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
    if (use_progressive_ && !seed_phase_complete_ && seed_expert_) {
      // During seed phase, rollback the seed expert
      seed_expert_->RollbackOneIter();
    } else {
      for (int k = 0; k < num_experts_; ++k) {
        experts_[k]->RollbackOneIter();
      }
      gate_->RollbackOneIter();
    }
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
  ComputeGateProbForInference(gate_raw.data(), gate_prob.data());

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
  ComputeGateProbForInference(gate_raw.data(), gate_prob.data());

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
  ComputeGateProbForInference(gate_raw.data(), gate_prob.data());

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
  ComputeGateProbForInference(gate_raw.data(), output);
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
  ComputeGateProbForInference(gate_raw.data(), gate_prob.data());

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
  ComputeGateProbForInference(gate_raw.data(), output);

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

void MixtureGBDT::GetResponsibilities(int64_t buffer_len, int64_t* out_len,
                                       double* out_data) const {
  // See header comment for the contract. Outside training (LoadModelFromString
  // path) num_data_ is 0 and responsibilities_ is empty — caller gets out_len=0
  // and an empty payload, which they can interpret as "no responsibilities to
  // snapshot here".
  *out_len = static_cast<int64_t>(responsibilities_.size());
  if (out_data == nullptr || buffer_len < *out_len || *out_len == 0) {
    return;  // size-query, undersized buffer, or nothing to copy
  }
  std::memcpy(out_data, responsibilities_.data(),
              static_cast<size_t>(*out_len) * sizeof(double));
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
  expert_pred_sm_.resize(nk);
  gate_proba_.resize(nk);
  gate_proba_no_bias_.assign(nk, 1.0 / num_experts_);
  yhat_.resize(num_data_);
  gradients_.resize(num_data_);
  hessians_.resize(num_data_);

  // Reset expert bias for loss-free load balancing
  std::fill(expert_bias_.begin(), expert_bias_.end(), 0.0);

  // Monotonicity baseline is invalidated when retraining on new data.
  prev_marginal_log_lik_ = -1e300;

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
  if (use_progressive_ && !seed_phase_complete_) {
    // During seed phase, only add to seed expert
    if (seed_expert_) {
      seed_expert_->AddValidDataset(valid_data, {});
    }
  } else {
    for (int k = 0; k < num_experts_; ++k) {
      experts_[k]->AddValidDataset(valid_data, {});  // No metrics at expert level
    }
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

  // Runtime-trained scalars used by the inference-time gate softmax. Without
  // these, a saved model loaded fresh would default to bias=0, T=1, var=1 and
  // route differently from the same model in-memory. See ComputeGateProbForInference.
  ss << "mixture_gate_temperature=" << gate_temperature_ << "\n";
  {
    ss << "mixture_expert_bias=";
    for (int k = 0; k < num_experts_; ++k) {
      if (k > 0) ss << ",";
      ss << expert_bias_[k];
    }
    ss << "\n";
  }
  {
    ss << "mixture_expert_variance=";
    for (int k = 0; k < num_experts_; ++k) {
      if (k > 0) ss << ",";
      ss << expert_variance_[k];
    }
    ss << "\n";
  }
  ss << "\n";

  // Gate model
  ss << "[gate_model]\n";
  ss << gate_->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
  ss << "\n";

  // Expert models
  if (use_progressive_ && !seed_phase_complete_ && seed_expert_) {
    // Still in seed phase - save seed expert as all experts
    for (int k = 0; k < num_experts_; ++k) {
      ss << "[expert_model_" << k << "]\n";
      ss << seed_expert_->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
      ss << "\n";
    }
  } else {
    for (int k = 0; k < num_experts_; ++k) {
      ss << "[expert_model_" << k << "]\n";
      ss << experts_[k]->SaveModelToString(start_iteration, num_iterations, feature_importance_type);
      ss << "\n";
    }
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

  // Restore runtime-trained gate scalars. Defaults match constructor / Init
  // values so models saved before this field was added still load correctly.
  expert_bias_.assign(num_experts_, 0.0);
  expert_variance_.assign(num_experts_, 1.0);
  gate_temperature_ = 1.0;
  if (params.count("mixture_gate_temperature")) {
    gate_temperature_ = std::stod(params["mixture_gate_temperature"]);
  }
  auto parse_csv_doubles = [&](const std::string& csv,
                               std::vector<double>* out, int expected) {
    std::stringstream sss(csv);
    std::string tok;
    int idx = 0;
    while (std::getline(sss, tok, ',') && idx < expected) {
      try {
        (*out)[idx] = std::stod(Common::Trim(tok));
      } catch (...) {
        (*out)[idx] = 0.0;
      }
      ++idx;
    }
  };
  if (params.count("mixture_expert_bias")) {
    parse_csv_doubles(params["mixture_expert_bias"], &expert_bias_, num_experts_);
  }
  if (params.count("mixture_expert_variance")) {
    parse_csv_doubles(params["mixture_expert_variance"], &expert_variance_,
                      num_experts_);
  }

  // Store loaded parameters for GetLoadedParam (must be valid JSON).
  // We classify a value as numeric only if it parses end-to-end as a number;
  // otherwise (e.g. comma-separated lists like `mixture_expert_bias`,
  // mode strings, anything with non-digit body) we quote it.
  auto is_full_number = [](const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != nullptr && *end == '\0';
  };
  std::stringstream param_ss;
  param_ss << "{";
  bool first = true;
  for (const auto& kv : params) {
    if (!first) {
      param_ss << ", ";
    }
    first = false;
    param_ss << "\"" << kv.first << "\": ";
    if (is_full_number(kv.second)) {
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

  // Restore training-time mixture parameters into the freshly-default config.
  // Without this block, every save/load round-trip silently snapped these
  // back to defaults — and `lgb.train` performs that round-trip *internally*
  // at the end of training (engine.py: `booster.model_from_string(model_to_string())`),
  // so even users who never explicitly save/load were affected.
  //
  // The most user-visible symptom was Markov inference: training with
  // `mixture_r_smoothing=markov` set `use_markov_=true` during fitting, but
  // after the post-train round-trip the loaded booster had `use_markov_=false`
  // and `mixture_smoothing_lambda=0`. `predict_markov()` then silently no-ops
  // (line 3341 / 3376 guards on `use_markov_`), and Python's
  // `predict_regime_proba_markov` no-ops too because it reads `lambda=0` /
  // `smoothing="none"` out of the loaded params dict. Test-set predictions
  // diverged from validation behaviour without any error or warning.
  if (params.count("mixture_r_smoothing")) {
    config_->mixture_r_smoothing = params["mixture_r_smoothing"];
  }
  if (params.count("mixture_smoothing_lambda")) {
    try {
      config_->mixture_smoothing_lambda =
          std::stod(params["mixture_smoothing_lambda"]);
    } catch (...) { /* keep default */ }
  }
  if (params.count("mixture_e_step_alpha")) {
    try {
      config_->mixture_e_step_alpha =
          std::stod(params["mixture_e_step_alpha"]);
    } catch (...) { /* keep default */ }
  }
  if (params.count("mixture_e_step_mode")) {
    config_->mixture_e_step_mode = params["mixture_e_step_mode"];
  }
  // Re-derive smoothing-mode flags from the restored config (mirrors Init()
  // line ~266). These flags are not serialized themselves; they are computed.
  use_markov_ = (config_->mixture_r_smoothing == "markov");
  use_momentum_ = (config_->mixture_r_smoothing == "momentum");

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
  if (use_progressive_ && !seed_phase_complete_ && seed_expert_) {
    total += seed_expert_->NumberOfTotalModel();
  } else {
    for (int k = 0; k < num_experts_; ++k) {
      total += experts_[k]->NumberOfTotalModel();
    }
    total += gate_->NumberOfTotalModel();
  }
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

int MixtureGBDT::GetNumLeavesForTree(int tree_idx) const {
  (void)tree_idx;
  Log::Fatal("MixtureGBDT::GetNumLeavesForTree is not implemented for mixture models");
  return 0;
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
