/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Mixture-of-Experts GBDT extension for regime-switching models.
 */
#ifndef LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_
#define LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_

#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "gbdt.h"

namespace LightGBM {

/*!
 * \brief Mixture-of-Experts GBDT implementation.
 *
 * This class implements a mixture-of-experts model where:
 * - K expert GBDTs specialize in different data regimes
 * - 1 gate GBDT determines which expert(s) to use for each sample
 * - Final prediction: yhat = sum_k(gate_k * expert_k)
 *
 * Training uses EM-style updates:
 * - E-step: Update responsibilities based on expert fit and gate probability
 * - M-step: Update experts with responsibility-weighted gradients
 * - M-step: Update gate with soft cross-entropy against full responsibilities
 *   (gradient = p_ik - r_ik), per Jordan-Jacobs hierarchical MoE.
 */
class MixtureGBDT : public GBDTBase {
 public:
  MixtureGBDT();
  ~MixtureGBDT();

  void Init(const Config* config, const Dataset* train_data,
            const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override;

  void MergeFrom(const Boosting* other) override;
  void ShuffleModels(int start_iter, int end_iter) override;
  void ResetTrainingData(const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override;
  void ResetConfig(const Config* config) override;
  void AddValidDataset(const Dataset* valid_data,
                       const std::vector<const Metric*>& valid_metrics) override;
  void Train(int snapshot_freq, const std::string& model_output_path) override;
  void RefitTree(const int* tree_leaf_prediction, const size_t nrow, const size_t ncol) override;

  /*!
   * \brief Training logic for one iteration (EM-style update)
   */
  bool TrainOneIter(const score_t* gradients, const score_t* hessians) override;

  void RollbackOneIter() override;
  int GetCurrentIteration() const override;
  std::vector<double> GetEvalAt(int data_idx) const override;
  const double* GetTrainingScore(int64_t* out_len) override;
  int64_t GetNumPredictAt(int data_idx) const override;
  void GetPredictAt(int data_idx, double* result, int64_t* out_len) override;
  int NumPredictOneRow(int start_iteration, int num_iteration, bool is_pred_leaf, bool is_pred_contrib) const override;

  void PredictRaw(const double* features, double* output,
                  const PredictionEarlyStopInstance* earlyStop) const override;
  void PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                       const PredictionEarlyStopInstance* early_stop) const override;
  void Predict(const double* features, double* output,
               const PredictionEarlyStopInstance* earlyStop) const override;
  void PredictByMap(const std::unordered_map<int, double>& features, double* output,
                    const PredictionEarlyStopInstance* early_stop) const override;
  void PredictLeafIndex(const double* features, double* output) const override;
  void PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const override;
  void PredictContrib(const double* features, double* output) const override;
  void PredictContribByMap(const std::unordered_map<int, double>& features,
                           std::vector<std::unordered_map<int, double>>* output) const override;

  std::string DumpModel(int start_iteration, int num_iteration, int feature_importance_type) const override;
  std::string ModelToIfElse(int num_iteration) const override;
  bool SaveModelToIfElse(int num_iteration, const char* filename) const override;
  bool SaveModelToFile(int start_iteration, int num_iterations, int feature_importance_type, const char* filename) const override;
  std::string SaveModelToString(int start_iteration, int num_iterations, int feature_importance_type) const override;
  bool LoadModelFromString(const char* buffer, size_t len) override;

  std::vector<double> FeatureImportance(int num_iteration, int importance_type) const override;
  double GetUpperBoundValue() const override;
  double GetLowerBoundValue() const override;
  int MaxFeatureIdx() const override;
  std::vector<std::string> FeatureNames() const override;
  int LabelIdx() const override;
  int NumberOfTotalModel() const override;
  int NumModelPerIteration() const override;
  int NumberOfClasses() const override;
  bool NeedAccuratePrediction() const override;
  void InitPredict(int start_iteration, int num_iteration, bool is_pred_contrib) override;

  double GetLeafValue(int tree_idx, int leaf_idx) const override;
  void SetLeafValue(int tree_idx, int leaf_idx, double val) override;
  int GetNumLeavesForTree(int tree_idx) const override;

  const char* SubModelName() const override { return "mixture"; }
  std::string GetLoadedParam() const override;
  std::string ParserConfigStr() const override;

  // MoE-specific prediction methods
  /*!
   * \brief Get regime (argmax of gate probabilities) for each sample
   * \param features Feature values
   * \param output Output array for regime indices
   */
  void PredictRegime(const double* features, int* output) const;

  /*!
   * \brief Get gate probabilities (regime probabilities) for each sample
   * \param features Feature values
   * \param output Output array of size K for probabilities
   */
  void PredictRegimeProba(const double* features, double* output) const;

  /*!
   * \brief Get individual expert predictions for each sample
   * \param features Feature values
   * \param output Output array of size K for expert predictions
   */
  void PredictExpertPred(const double* features, double* output) const;

  /*!
   * \brief Get number of experts
   */
  int NumExperts() const { return num_experts_; }

  /*!
   * \brief Number of training rows the responsibilities buffer is sized for.
   * Returns 0 outside of training (e.g. on a freshly loaded model).
   */
  int NumTrainData() const { return num_data_; }

  /*!
   * \brief Copy the current training-time responsibilities r_ik into a caller
   * buffer in sample-major layout (`out[i * K + k]`).
   *
   * Intended for diagnostic use during training — typically from a
   * `lgb.train(callbacks=[...])` callback that snapshots r at each iteration.
   * What "current" means depends on the callback timing relative to the EM
   * loop:
   *   - Iter 0..warmup_iters-1: r holds the InitResponsibilities output
   *     (e.g. GMM / kmeans / quantile clustering of the labels). The E-step
   *     does not run during warmup, so this is the *initialization* you can
   *     plot as the "before EM" baseline.
   *   - Iter >= warmup_iters: r is the latest E-step posterior, recomputed
   *     each iteration from the current expert predictions and gate prior.
   *
   * Behavior outside training: `responsibilities_` is not serialized, so a
   * model loaded via LoadModelFromString returns an empty buffer (out_len=0)
   * even though its experts and gate are intact. Use this only on a booster
   * still attached to its training data.
   *
   * Two-call protocol matching `LGBM_BoosterGetLoadedParam`:
   *   1. Pass `out_data=nullptr` (or buffer_len=0) to read the required size
   *      into `out_len`.
   *   2. Allocate a `double[out_len]` buffer and call again with
   *      `buffer_len >= out_len`.
   * If `buffer_len < out_len`, no copy is performed and the caller should
   * reallocate.
   *
   * \param buffer_len Capacity of `out_data` in number of doubles.
   * \param out_len    [out] Required buffer size = num_data * num_experts.
   * \param out_data   [out] Destination buffer (may be nullptr on size query).
   */
  void GetResponsibilities(int64_t buffer_len, int64_t* out_len,
                           double* out_data) const;

  /*!
   * \brief Check if Markov mode is enabled
   */
  bool IsMarkovMode() const { return use_markov_; }

  // Markov-specific prediction methods (use previous gate probabilities)
  /*!
   * \brief Predict with previous gate probabilities (for Markov mode time-series inference)
   * \param features Original feature values
   * \param prev_proba Previous gate probabilities (size K), nullptr for uniform
   * \param output Output prediction
   */
  void PredictWithPrevProba(const double* features, const double* prev_proba, double* output) const;

  /*!
   * \brief Get gate probabilities with previous proba (for Markov mode)
   * \param features Original feature values
   * \param prev_proba Previous gate probabilities (size K), nullptr for uniform
   * \param output Output array of size K for probabilities
   */
  void PredictRegimeProbaWithPrevProba(const double* features, const double* prev_proba, double* output) const;

 protected:
  /*!
   * \brief Initialize expert responsibilities (uniform, kmeans, etc.)
   */
  void InitResponsibilities();

  /*!
   * \brief Initialize responsibilities using Balanced K-Means.
   * \param labels Label array
   * \param include_label If true, label is concatenated as an extra feature
   *   dimension (legacy "balanced_kmeans" behavior — biases clusters toward
   *   y-magnitude). If false, clustering uses raw features only — proper
   *   regime discovery in X-space.
   */
  void InitResponsibilitiesBalancedKMeans(const label_t* labels, bool include_label);

  /*!
   * \brief Initialize responsibilities using GMM.
   * \param labels Label array
   * \param include_label If true, label is included as an extra dimension
   *   (legacy "gmm" behavior). If false, GMM is fit on features only.
   */
  void InitResponsibilitiesGMM(const label_t* labels, bool include_label);

  /*!
   * \brief Initialize responsibilities using tree-based hierarchical clustering
   * \param labels Label array
   *
   * Algorithm:
   * 1. Train a deep decision tree to predict y from features
   * 2. Get leaf index for each sample
   * 3. Compute mean y for each leaf
   * 4. Build distance matrix between leaves (|mean_y_i - mean_y_j|)
   * 5. Hierarchical clustering (agglomerative) to merge leaves into K groups
   * 6. Assign samples to experts based on their leaf's cluster
   */
  void InitResponsibilitiesTreeHierarchical(const label_t* labels);

  /*!
   * \brief Forward pass: compute expert predictions and gate probabilities
   */
  void Forward();

  /*!
   * \brief E-step: update responsibilities based on expert fit and gate probability
   */
  void EStep();

  /*!
   * \brief Update per-expert noise scale (σ_k² for L2, b_k for L1) from
   * responsibility-weighted residuals. Required for proper Gaussian/Laplace
   * MoE EM (Jordan-Jacobs). No-op when mixture_estimate_variance is false.
   */
  void UpdateExpertVariances();

  /*!
   * \brief Compute training-set marginal log-likelihood (ELBO):
   *     Σ_i log Σ_k π_k(x_i) p(y_i | x_i, f_k, σ_k²)
   * Returns -inf if the model isn't fit yet. Only meaningful when variances
   * are estimated.
   */
  double ComputeMarginalLogLikelihood() const;

  /*!
   * \brief Apply time-series smoothing to responsibilities (EMA or Markov)
   */
  void SmoothResponsibilities();

  /*!
   * \brief M-step for experts: update with responsibility-weighted gradients
   */
  void MStepExperts();

  /*!
   * \brief M-step for gate: soft CE against full responsibilities (p - r)
   */
  void MStepGate();

  /*!
   * \brief v0.7 leaf-refit pass: rewrite leaf values of all existing expert
   *        trees AND gate trees against the current responsibilities, before
   *        appending the next tree. Restores classical-EM "free-parameter"
   *        behavior on the closed-form M-step over each tree's existing
   *        partition structure (issue #37).
   *
   * Called from TrainOneIter between the E-step and MStepExperts when
   * `mixture_refit_leaves=true` AND ShouldRefit() fires. After refit,
   * Forward() is re-run so MStepExperts / MStepGate see the post-refit
   * expert_pred_ / gate_proba_.
   *
   * For each expert k: builds an r-weighted gradient callback and calls
   * experts_[k]->RefitLeavesByGradients. For the gate: builds the soft-CE
   * gradient callback (mirroring MStepGate's gradient form including Friedman
   * K/(K-1), temperature chain rule, and Dirichlet shrinkage) and calls
   * gate_->RefitLeavesByGradients. No-op when gate_type is "leaf_reuse" or
   * "none" since those do not own a refittable gate GBDT.
   */
  void RefitExpertsAndGate();

  /*!
   * \brief v0.7 leaf-refit trigger gate. Decides whether the current EM
   *        round should run RefitExpertsAndGate. Always returns false
   *        when `mixture_refit_leaves` is off; otherwise dispatches on
   *        `mixture_refit_trigger`:
   *
   *   - "always": fires every post-warmup iter (highest cost, most faithful EM)
   *   - "elbo":   fires when the most recent ELBO log showed a >5% drop —
   *               cheap because ELBO is only computed every 10 iters and
   *               this reads `last_elbo_drop_ratio_` set by that block
   *   - "every_n": fires every `mixture_refit_every_n` post-warmup iters
   */
  bool ShouldRefit() const;

  /*!
   * \brief M-step for gate using leaf-reuse routing.
   * Derives gate probabilities from expert tree leaf statistics
   * and periodically retrains gate GBDT for inference.
   */
  void MStepGateLeafReuse();

  /*!
   * \brief Compute pointwise loss for E-step
   */
  double ComputePointwiseLoss(double y, double pred) const;

  /*!
   * \brief Numerically stable softmax
   */
  void Softmax(const double* scores, int n, double* probs) const;

  /*!
   * \brief Apply softmax to gate raw scores using the inference-time pipeline:
   * `softmax((gate_raw + expert_bias_) / gate_temperature_)`.
   *
   * Forward()/ForwardValid() apply bias and temperature when computing the
   * routing distribution at training/validation time, so any model trained
   * with non-default `mixture_balance_factor` (which moves expert_bias_) or
   * temperature annealing (`mixture_gate_temperature_*`) produces a routing
   * that depends on those scalars. Predict* paths must apply the same
   * transformation or test-time routing silently diverges from training.
   */
  void ComputeGateProbForInference(const double* gate_raw, double* gate_prob) const;

  /*! \brief Number of experts (K) */
  int num_experts_;

  /*! \brief Expert GBDTs */
  std::vector<std::unique_ptr<GBDT>> experts_;

  /*! \brief Gate GBDT (multiclass with K classes) */
  std::unique_ptr<GBDT> gate_;

  /*! \brief Base config for experts (shared settings) */
  std::unique_ptr<Config> expert_config_;

  /*! \brief Per-expert configs (with different seeds for symmetry breaking) */
  std::vector<std::unique_ptr<Config>> expert_configs_;

  /*! \brief Config for gate */
  std::unique_ptr<Config> gate_config_;

  /*! \brief Original config */
  std::unique_ptr<Config> config_;

  /*! \brief Responsibilities r_ik (N x K) */
  std::vector<double> responsibilities_;

  /*! \brief Expert predictions in expert-major layout (K x N): expert_pred_[k*N + i] */
  std::vector<double> expert_pred_;

  /*! \brief Expert predictions in sample-major layout (N x K): expert_pred_sm_[i*K + k] */
  std::vector<double> expert_pred_sm_;

  /*! \brief Gate probabilities (N x K). For "gbdt" gate type this is
   *  softmax((z + expert_bias_) / T) — i.e. the bias-included routing used
   *  for forward yhat, validation metrics, and inference. */
  std::vector<double> gate_proba_;

  /*! \brief Bias-free gate probabilities (N x K): softmax(z / T) without
   *  expert_bias_ added. Read by the E-step prior, ELBO, and
   *  ComputeAffinityScores so that the load-balancing bias acts only on the
   *  routing decision (forward pass + inference), not on the probabilistic
   *  model that defines responsibilities. Without this split, the soft-CE
   *  target r in MStepGate would be computed from a bias-included prior, so
   *  the gate would still have to learn to undo the bias each iter — partly
   *  defeating the DeepSeek "Auxiliary-Loss-Free Load Balancing" design that
   *  PR #25 aimed to implement on the gradient side.
   *
   *  For "none" / "leaf_reuse" gate types (no bias is ever applied to
   *  gate_proba_ in those modes) this buffer is just a copy of gate_proba_. */
  std::vector<double> gate_proba_no_bias_;

  /*! \brief Combined prediction yhat (N) */
  std::vector<double> yhat_;

  /*! \brief Last computed marginal log-likelihood, for monotonicity diagnostic.
   *  EM with an exact M-step is non-decreasing here; the GBDT M-step is only
   *  approximate so small drops are normal, but persistent / large drops
   *  indicate misalignment (dropout, adaptive_lr, aggressive annealing, or a
   *  re-introduced math bug). Negative-infinity sentinel for "not yet seen".
   */
  double prev_marginal_log_lik_ = -1e300;

  /*! \brief Last seen ELBO drop ratio (positive = drop, negative = improvement),
   *  normalized by `max(|prev_marginal_log_lik_|, 1.0)`. Updated by the
   *  every-10-iter ELBO log block in TrainOneIter. Read by ShouldRefit() in
   *  "elbo" trigger mode — refit fires when this exceeds 0.05.
   *  Stays at 0.0 when `mixture_estimate_variance=false` (no ELBO computed).
   */
  double last_elbo_drop_ratio_ = 0.0;

  /*! \brief Gradients for mixture (N) */
  std::vector<score_t> gradients_;

  /*! \brief Hessians for mixture (N) */
  std::vector<score_t> hessians_;

  /*! \brief Training data */
  const Dataset* train_data_;

  /*! \brief Objective function */
  const ObjectiveFunction* objective_function_;

  /*! \brief Training metrics */
  std::vector<const Metric*> training_metrics_;

  /*! \brief Number of data points */
  data_size_t num_data_;

  /*! \brief Current iteration */
  int iter_;

  /*! \brief Max feature index */
  int max_feature_idx_;

  /*! \brief Feature names */
  std::vector<std::string> feature_names_;

  /*! \brief Label index */
  int label_idx_;

  /*! \brief E-step loss type (l2, l1, quantile) */
  std::string e_step_loss_type_;

  /*! \brief Per-expert noise scale (size K).
   *
   * Interpretation depends on e_step_loss_type_:
   *   - "l2"      → variance σ_k² (Gaussian likelihood)
   *   - "l1"      → Laplace scale b_k
   *   - "quantile"→ pseudo-scale carrying the same role as σ_k² (no proper density)
   *
   * Updated each iter from responsibility-weighted residuals when
   * mixture_estimate_variance=true. Initialized to a sensible default
   * (overall residual scale / K) and floored with kMixtureVarianceFloor to
   * prevent collapse.
   */
  std::vector<double> expert_variance_;

  /*! \brief Loaded parameter string for serialization */
  std::string loaded_parameter_;

  // Markov switching members
  /*! \brief Whether Markov mode is enabled */
  bool use_markov_;

  /*! \brief Previous gate probabilities for Markov mode (N x K) */
  std::vector<double> prev_gate_proba_;

  /*! \brief Whether momentum mode is enabled */
  bool use_momentum_;

  // Loss-free load balancing members
  /*! \brief Expert bias for load balancing (size K) */
  std::vector<double> expert_bias_;

  /*! \brief Expert load from previous iteration (size K), used for auxiliary loss */
  std::vector<double> expert_load_;

  /*! \brief Update expert bias based on recent load */
  void UpdateExpertBias();

  /*! \brief Update expert load from current responsibilities */
  void UpdateExpertLoad();

  // Expert dropout members
  /*! \brief Random number generator for expert dropout */
  mutable std::mt19937 dropout_rng_;

  /*! \brief Uniform distribution for dropout probability */
  mutable std::uniform_real_distribution<double> dropout_dist_;

  // Sparse activation buffers (#10) — held as members so the pointers passed
  // to GBDT::SetBaggingData remain valid for the entire lifetime of each
  // expert's data_partition_->used_data_indices_, not just the duration of a
  // single MStepExperts call. Using stack-allocated buffers here caused
  // segfaults (#16) when the same indices were re-read on subsequent
  // BeforeTrain() calls.
  std::vector<std::vector<data_size_t>> expert_sample_indices_;
  std::vector<data_size_t> all_data_indices_;

  // MStepGateLeafReuse scratch / cache (#16 perf follow-up).
  // - leaf_reuse_iters_: BinIterator* indexed by inner_feat for O(1) lookup.
  //   Replaces the per-thread std::unordered_map that was the hot-path
  //   bottleneck at 300+ Optuna trials with deep trees.
  // - leaf_reuse_iter_features_: tracks which slots in leaf_reuse_iters_
  //   currently own a heap-allocated iterator (for cleanup on tree-shape change).
  // - sample_leaf_buf_, leaf_expert_sum_buf_, leaf_count_buf_: per-iteration
  //   scratch buffers, retained across calls to avoid heap churn.
  std::vector<class BinIterator*> leaf_reuse_iters_;
  std::vector<int> leaf_reuse_iter_features_;
  std::vector<int> sample_leaf_buf_;
  std::vector<double> leaf_expert_sum_buf_;
  std::vector<int> leaf_count_buf_;

  // Adaptive per-expert learning rate members
  /*! \brief Per-expert loss history (ring buffer per expert) */
  std::vector<std::vector<double>> expert_loss_history_;

  /*! \brief Current write position in loss history ring buffer */
  int loss_history_pos_ = 0;

  /*! \brief Per-expert LR multiplier (computed from loss trend) */
  std::vector<double> expert_lr_scale_;

  // Expert Choice Routing members
  /*! \brief Expert capacity (samples per expert) */
  int expert_capacity_;

  /*! \brief Whether to use expert choice routing */
  bool use_expert_choice_;

  /*! \brief Affinity scores for expert choice (N x K, sample-major) */
  std::vector<double> affinity_scores_;

  /*! \brief Selection mask for expert choice (N x K, sample-major) */
  std::vector<int> expert_selection_mask_;

  /*!
   * \brief E-step using Expert Choice Routing
   */
  void EStepExpertChoice();

  /*!
   * \brief Compute affinity scores for expert choice
   */
  void ComputeAffinityScores();

  /*!
   * \brief Each expert selects top-C samples
   */
  void SelectTopSamplesPerExpert();

  /*!
   * \brief Convert selection mask to soft responsibilities
   */
  void ConvertSelectionToResponsibilities();

  /*!
   * \brief Spawn K experts from a trained seed GBDT (EvoMoE progressive training)
   */
  void SpawnExpertsFromSeed();

  /*!
   * \brief Compute current gate temperature based on MoE iteration progress
   * \param moe_iter Current iteration within MoE phase
   * \param total_moe_iters Total number of MoE phase iterations
   * \return Current temperature value
   */
  double ComputeTemperature(int moe_iter, int total_moe_iters) const;

  /*!
   * \brief If `responsibilities_` is essentially uniform after Init, inject a
   * deterministic per-(sample, expert) perturbation so EM has something to grip
   * on. Without this, uniform r is a fixed point of the EM iteration: every
   * expert sees the same gradient → every expert builds the same tree → r
   * stays uniform forever. The trap was empirically confirmed in
   * `examples/em_init_sensitivity.py` (uniform-init final entropy stayed
   * pinned at log(K), and no combination of `hard_m_step` /
   * `mixture_estimate_variance` / `mixture_diversity_lambda` could break it).
   * No-op when r is already non-uniform — non-uniform inits (gmm, kmeans,
   * tree_hierarchical, etc.) reach the early-return without modification.
   */
  void BreakUniformSymmetryIfNeeded();

  /*!
   * \brief Forward pass for validation data: compute expert predictions and gate probabilities
   * \param valid_idx Index of validation dataset
   */
  void ForwardValid(int valid_idx);

  /*!
   * \brief Output metrics and check early stopping
   * \param iter Current iteration
   * \return Non-empty string if early stopping triggered
   */
  std::string OutputMetric(int iter);

  /*!
   * \brief Evaluate metrics and check early stopping condition
   * \return true if early stopping triggered
   */
  bool EvalAndCheckEarlyStopping();

  // ===== Validation data storage =====
  /*! \brief Validation datasets */
  std::vector<const Dataset*> valid_datas_;

  /*! \brief Metrics for each validation dataset */
  std::vector<std::vector<const Metric*>> valid_metrics_;

  // ===== Validation prediction buffers =====
  /*! \brief Expert predictions for validation (expert-major: [k * num_valid + i]) */
  std::vector<std::vector<double>> expert_pred_valid_;

  /*! \brief Gate probabilities for validation (sample-major: [i * K + k]) */
  std::vector<std::vector<double>> gate_proba_valid_;

  /*! \brief Combined predictions for validation */
  std::vector<std::vector<double>> yhat_valid_;

  /*! \brief Previous gate probabilities for Markov mode validation */
  std::vector<std::vector<double>> prev_gate_proba_valid_;

  // ===== Progressive training (EvoMoE) =====
  /*! \brief Whether progressive mode is enabled */
  bool use_progressive_;

  /*! \brief Number of seed training iterations */
  int seed_iterations_;

  /*! \brief Seed GBDT for progressive training */
  std::unique_ptr<GBDT> seed_expert_;

  /*! \brief Whether seed phase has finished */
  bool seed_phase_complete_;

  // ===== Gate temperature annealing =====
  /*! \brief Current gate temperature */
  double gate_temperature_;

  // ===== Early stopping =====
  /*! \brief Number of rounds for early stopping (0 = disabled) */
  int early_stopping_round_;

  /*! \brief Minimum improvement for early stopping */
  double early_stopping_min_delta_;

  /*! \brief Only use first metric for early stopping */
  bool es_first_metric_only_;

  /*! \brief Best iteration for each validation dataset and metric */
  std::vector<std::vector<int>> best_iter_;

  /*! \brief Best score for each validation dataset and metric */
  std::vector<std::vector<double>> best_score_;

  /*! \brief Output message of best iteration */
  std::vector<std::vector<std::string>> best_msg_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_SRC_BOOSTING_MIXTURE_GBDT_H_
