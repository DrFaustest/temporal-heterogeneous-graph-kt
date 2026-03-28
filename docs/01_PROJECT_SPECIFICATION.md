# THGKT Project Specification

## Project Title
**Temporal Heterogeneous Graph Neural Networks for Knowledge Tracing and Concept Dependency Modeling**

## Project Type
Ambitious application/algorithmic deep learning project.

## Research Goal
Develop a deep learning system that predicts whether a student will answer the next question correctly by combining:

1. **Heterogeneous graph neural modeling** of students, questions, and concepts
2. **Temporal sequence modeling** of student interaction histories

## Core Research Question
Does a temporal heterogeneous graph neural architecture outperform standard sequential knowledge tracing baselines while also offering improved interpretability through concept- and prerequisite-aware reasoning?

## Problem Definition
Given a student's historical interaction sequence with educational content, predict the probability of correctness for the next question. The system should also model concept dependencies and provide interpretable insight into which concepts and relationships influenced the prediction.

## Primary Task
**Next-response prediction**

Predict:

```math
P(r_{t+1}=1 \mid \text{past student interactions}, q_{t+1}, c_{t+1}, G)
```

where:
- `r_(t+1)` is correctness of the next response
- `q_(t+1)` is the next question
- `c_(t+1)` is the concept or concept set associated with that question
- `G` is the heterogeneous educational graph

## Secondary Goals
- Estimate concept mastery over time
- Model concept prerequisite structure
- Support explainability through concept importance and prerequisite influence
- Compare graph-based, sequence-based, and hybrid models

## Target Datasets

### Primary Dataset
**ASSISTments-style knowledge tracing dataset**

Reason:
- manageable scale
- widely used benchmark
- suitable for fast iteration and debugging

### Secondary Dataset
**EdNet**

Reason:
- larger scale
- richer interaction history
- useful for validating scalability after the core system works

## Model Scope

### Inputs
- Student interaction logs
- Question-to-concept mappings
- Time or sequence order
- Correctness labels
- Optional time lag, attempt count, elapsed time
- Derived or provided concept prerequisite graph

### Outputs
- Probability of correctness for next student response
- Optional concept mastery scores
- Optional importance/attention summaries over graph relations or concepts

## Baselines Required
The codebase must support comparison against at least:
1. **Logistic regression or MLP baseline**
2. **DKT-style recurrent model**
3. **Graph-only model**
4. **Full temporal heterogeneous graph model**

## Main Proposed Model
A hybrid model with:
1. **Heterogeneous graph encoder**
2. **Temporal student-state encoder**
3. **Prediction head**
4. Optional explanation output heads

## Required Evaluation Metrics
- AUC
- Accuracy
- F1 score
- Binary cross-entropy loss

Optional if practical:
- Precision
- Recall
- Brier score
- Calibration error

## Required Experiment Types
1. Main model comparison
2. Ablation study
3. Graph construction comparison
4. Optional explainability case study

## Key Constraints
- No temporal leakage
- Reproducible splits
- Config-driven experiments
- Dataset adapters instead of hardcoded dataset-specific logic
- Start with ASSISTments-compatible small-scale execution before scaling

## Software Requirements
- Python 3.11+
- PyTorch
- PyTorch Geometric
- YAML configuration support
- Modular package organization
- Typed interfaces where practical
- Unit and integration tests
- Saved artifacts for processed data and experiment outputs
