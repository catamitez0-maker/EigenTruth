# EigenTruth Methodology

EigenTruth is a research toolkit for monitoring representation drift and testing
activation steering interventions in decoder-only language models.

It is not a production hallucination detector and does not prove that a model
output is true.

## Core Idea

EigenTruth treats hallucination-prone generation as a potential geometric
deviation in hidden-state flow.

The current implementation builds a truth manifold from factual warmup samples,
then monitors the target layer's last-token hidden state during generation.
When the hidden state moves far from the warmup manifold, EigenTruth can inject
a steering vector toward a factual centroid or a contrastive truth direction.

## Pipeline

1. Collect hidden states from a target Transformer layer on factual examples.
2. Incrementally build a `TruthManifold` using an online mean and a regularized
   inverse-scatter precision proxy.
3. Optionally collect false examples to build a contrastive direction.
4. Register a PyTorch `forward_hook` on the selected layer.
5. During generation, compute:
   - Mahalanobis distance from the truth manifold
   - Poincare projection of hidden states
   - Hyperbolic Semantic Entropy over a sliding window
6. If distance exceeds the configured threshold, optionally inject a normalized
   steering vector into the last-token activation.

## What EigenTruth Measures

EigenTruth measures geometry in representation space:

- distance from a warmup manifold
- dispersion of recent hidden states in a hyperbolic projection
- whether a configured intervention changes generation trajectories

These diagnostics are useful for experiments, ablations, and qualitative demos.

## Distance Calibration

The current `cov_inv` field is a Sherman-Morrison online precision proxy over
warmup-state deviations. It is designed for fast monitoring and relative
threshold sweeps, not as an exact inverse of the empirical sample covariance.

Treat Mahalanobis thresholds as experiment-specific hyperparameters. Calibrate
them per model, target layer, warmup set, and generation setup before comparing
results.

## What EigenTruth Does Not Prove

EigenTruth does not prove:

- that an output is factually true
- that a model is safe
- that a hallucination has been eliminated
- that a correction is semantically valid
- that one model is globally better than another

The current metrics are representation diagnostics, not truth labels.

## Experimental Use

Use EigenTruth for controlled experiments:

- compare with and without steering on the same prompts
- sweep target layers and thresholds
- compare factual and false warmup sets
- record output differences, distance, and HSE
- evaluate outputs with external benchmarks or human review

Recommended external evaluation:

- TruthfulQA
- HaluEval
- FEVER-style factuality tasks
- task-specific human evaluation

## Limitations

- Results depend strongly on the target layer.
- Warmup data quality matters.
- Small warmup sets can create fragile manifolds.
- Mahalanobis thresholds are not portable across models or layers without
  calibration.
- Steering can change wording without improving truthfulness.
- A lower Mahalanobis distance does not guarantee factual correctness.
- HSE is an experimental dispersion signal, not a calibrated risk score.
- The current examples are demonstrations, not benchmark evidence.

## Current Status

EigenTruth is a research preview. The codebase has unit tests for numerical
stability, hook behavior, warmup, diagnostics, and wrapper lifecycle, but it
still needs standardized benchmark evaluation before any production claims.
