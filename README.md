# THGKT

Temporal Heterogeneous Graph Neural Networks for Knowledge Tracing and Concept Dependency Modeling.

## What This Project Does

This project predicts whether a student will answer the next question correctly.

It does that by combining two views of the same learning process:

1. A **heterogeneous graph** over:
- students
- questions
- concepts
- concept prerequisite relations

2. A **temporal interaction sequence** for each student:
- which questions they saw
- whether they were correct
- elapsed time
- attempt count
- the concepts attached to each question

The core research question is whether next-response prediction improves when both views are used together instead of using only a sequence model or only a graph model.

The repository includes:
- a canonical multi-table schema for educational interaction data
- dataset adapters for ASSISTments-style data and EdNet KT1
- leakage-safe preprocessing and chronological splitting
- graph construction and sequence construction pipelines
- baseline models
- the full THGKT model in PyTorch and PyTorch Geometric
- training, evaluation, checkpointing, plotting, and comparison tooling

## Contributions

This project is organized around four concrete implementation components:

- a unified canonical pipeline for educational interaction data that supports both ASSISTments-style data and EdNet while keeping dataset-specific logic isolated inside adapters
- a hybrid THGKT architecture that fuses heterogeneous structural context with temporal student interaction modeling for next-response prediction
- configurable concept-relation construction strategies that allow prerequisite-style, transition-based, and co-occurrence-based graph variants to be compared under the same training framework
- a reproducible ablation benchmark suite with saved artifacts, config snapshots, checkpoints, plots, and comparison reports for fair model evaluation

The repository is therefore not framed as a novelty claim about a new class of knowledge tracing model. It provides an implementation and evaluation workflow for testing whether heterogeneous graph structure adds predictive value beyond standard sequence-based knowledge tracing in this project setting.

## Relation To Prior Knowledge Tracing Models

This project sits in the line of work that starts with sequence-based knowledge tracing models such as DKT, extends through memory-based approaches such as DKVMN, and later incorporates richer relational structure through graph-based methods such as GKT and heterogeneous graph KT variants. Transformer-based KT models further improve sequential representation power, but they are still primarily temporal models unless structural relations are explicitly encoded.

Relative to that landscape, THGKT is positioned as a fusion model. DKT is the sequence-only baseline in this repository. Graph-only baselines test whether educational structure alone is useful. The full THGKT model combines both views by jointly modeling a heterogeneous student-question-concept graph and a temporal interaction encoder.

Prior graph-based KT work shows that structural relations can improve prediction, but many graph formulations focus mainly on concept-level or question-level connectivity. The goal here is to make that structural view more explicit by using typed student, question, and concept nodes together with configurable concept relation generation, then testing those choices through controlled ablations.

## Why Graph Structure Should Help

Knowledge tracing is not only a sequence problem. It is also a dependency-structure problem.

A student answers questions over time, but those questions are attached to concepts, and concepts often have prerequisite relationships. If a learner improves on fraction addition, that signal should inform nearby concepts such as equivalent fractions or mixed-number operations when the curriculum encodes those dependencies. A sequence-only model can observe temporal co-occurrence, but it does not explicitly represent the educational structure that ties those observations together.

The graph inductive bias in THGKT is intended to model three assumptions that are standard in concept-based tutoring systems:
- student state is partially reflected by the concepts attached to the questions they answer
- questions are not independent items; they are evidence about latent concept mastery
- concept mastery may propagate locally across prerequisite neighborhoods

In that view, graph message passing is useful because it lets the model move information along typed edges such as `student -> question -> concept` and `concept -> concept`. That gives the model a structural mechanism for propagating evidence across prerequisite neighborhoods instead of requiring the temporal encoder to infer the entire dependency pattern implicitly from token order alone.

Stated more directly: graph message passing allows propagation of mastery signals across prerequisite concept neighborhoods, which sequence-only models cannot represent structurally.

## What The Model Predicts

For each target interaction, the model predicts:
- `P(correct_next = 1 | student history, question, concepts, graph context)`

The main evaluation metrics are:
- AUC
- Accuracy
- F1
- BCE loss

## Research Framing

The repository is built around a set of testable hypotheses rather than a single monolithic model claim.

- `H1`: adding graph structure improves next-response prediction over sequence-only knowledge tracing because student-question-concept relations encode cross-item dependency information.
- `H2`: adding temporal encoding improves prediction over graph-only modeling because recent order, recency, and response history carry short-term knowledge-state information not recoverable from the static graph alone.
- `H3`: different concept-relation construction strategies (prerequisite, transition, co-occurrence) produce measurable differences in predictive performance.
- `H4`: explicitly defined prerequisite edges may not outperform data-driven relation construction methods.

The benchmark suite is organized around these hypotheses:
- DKT baseline tests the sequence-only view
- graph-only baseline tests the structure-only view
- THGKT full tests the joint view
- THGKT without prerequisite edges tests whether concept dependency information matters
- THGKT without the temporal encoder tests whether sequential dynamics matter once graph structure is present
- THGKT with an alternative relation generator tests whether graph construction strategy changes predictive quality

## Experimental Results Summary

Across the EdNet-1k benchmark suite, THGKT achieves the highest AUC,
with the best configuration (co-occurrence graph) reaching approximately 0.7275.

However, improvements over the DKT baseline are modest:
- THGKT (best): ~0.7275 AUC
- DKT baseline: ~0.7243 AUC

This indicates that graph-enhanced temporal modeling provides consistent but incremental gains in predictive performance.

Ablation results show:

- Removing the temporal encoder significantly degrades performance
- Removing prerequisite edges does not reduce performance and may slightly improve it
- Co-occurrence-based graph construction performs slightly better than transition-based prerequisite graphs

These results suggest that:
- Temporal modeling remains the dominant signal
- Graph structure provides auxiliary improvements
- Explicit prerequisite edges may not reflect true learning dependencies in the dataset

However, empirical results in this repository suggest that while graph structure provides useful inductive bias,
its contribution is secondary to temporal modeling. In particular, data-driven relation construction
(e.g., co-occurrence) appears more effective than manually specified prerequisite edges,
indicating that learned statistical relationships may better capture dependencies than explicit curriculum assumptions.

## Key Findings

- Temporal sequence modeling is the dominant factor in predictive performance
- Graph augmentation provides consistent but modest improvements in AUC
- Co-occurrence-based concept relations outperform prerequisite-based graphs
- Explicit prerequisite edges do not improve performance and may introduce noise
- Graph-only models perform significantly worse, confirming that structure alone is insufficient

## Metric Interpretation

Different models optimize different aspects of prediction quality:

- AUC reflects ranking quality and probabilistic discrimination
- Accuracy and F1 depend on thresholded predictions

In this benchmark:
- THGKT achieves the best AUC, indicating improved ranking of correct vs incorrect responses
- DKT achieves slightly higher accuracy and F1, suggesting stronger calibration for binary classification

This highlights a tradeoff between probabilistic modeling quality and classification threshold performance.

## Experimental Design Notes

All results are reported across multiple random seeds to account for stochastic training variation.
Observed improvements should therefore be interpreted as consistent trends rather than single-run artifacts.

## Where To Find Results

See:
- `artifacts/.../comparison_metrics.csv`
- `comparison/comparison_report.md`

for full benchmark outputs, including AUC, accuracy, F1, and BCE loss across all variants.

## How The Project Works

The pipeline is intentionally modular.

### 1. Raw Educational Data Is Converted Into A Canonical Schema

All dataset-specific logic stays inside dataset adapters. The rest of the project works only on the canonical schema.

The canonical bundle contains five tables:
- `interactions`
- `questions`
- `concepts`
- `question_concept_map`
- `concept_relations`

The canonical interaction row includes:
- `student_id`
- `interaction_id`
- `seq_idx`
- `timestamp`
- `question_id`
- `correct`
- `concept_ids`
- `elapsed_time`
- `attempt_count`
- `source_dataset`

Example:

```python
from pathlib import Path

from thgkt.data.adapters.ednet import EdNetAdapter, EdNetAdapterConfig
from thgkt.schemas.validators import validate_bundle

adapter = EdNetAdapter(EdNetAdapterConfig(max_users=20000))
raw = adapter.load_raw(
    Path("data/raw/ednet/KT1"),
    contents_path=Path("data/raw/ednet/contents"),
)
bundle = adapter.to_canonical(raw)
report = validate_bundle(bundle)

print(bundle.table_sizes())
print(report)
```

Relevant code:
- [canonical.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/schemas/canonical.py)
- [validators.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/schemas/validators.py)
- [assistments.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/adapters/assistments.py)
- [ednet.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/adapters/ednet.py)

### 2. The Canonical Bundle Is Cleaned And Split Without Temporal Leakage

The preprocessing stage:
- removes invalid rows
- normalizes IDs and value types
- filters students with too few interactions
- recomputes sequence indices per student

The split stage is chronological and leakage-safe.

Example:

```python
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.splitting import SplitConfig, make_splits

preprocessor = CanonicalPreprocessor(
    PreprocessingConfig(
        min_interactions_per_student=4,
        drop_interactions_without_concepts=True,
    )
)
clean_bundle = preprocessor.run(bundle)

splits = make_splits(
    clean_bundle,
    SplitConfig(
        strategy="student_chronological",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    ),
)
```

Relevant code:
- [preprocessing.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/preprocessing.py)
- [splitting.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/splitting.py)

### 3. The Same Data Is Projected Into A Heterogeneous Graph

Node types:
- `student`
- `question`
- `concept`

Edge types:
- `student answered question`
- `question tests concept`
- `concept prerequisite_of concept`
- optional `student exposure_to concept`

Example:

```python
from thgkt.graph import GraphBuilderConfig, build_hetero_graph

graph = build_hetero_graph(
    clean_bundle,
    splits,
    GraphBuilderConfig(
        interaction_edge_split="train",
        include_student_exposure_edges=True,
        include_reverse_edges=False,
    ),
)

print(graph.node_counts())
print(graph.edge_counts())
```

Relevant code:
- [builder.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/graph/builder.py)
- [pyg.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/graph/pyg.py)

### 4. The Same Data Is Also Projected Into Next-Response Sequences

For every target interaction, the sequence pipeline builds a prefix history and a target question label.

This repository uses a shared per-student history representation so large runs do not duplicate the same history prefix for every example.

Example:

```python
from thgkt.sequences import SequenceBuilderConfig, build_sequence_artifacts

sequences = build_sequence_artifacts(
    clean_bundle,
    splits,
    SequenceBuilderConfig(
        allow_empty_history=False,
        max_history_length=128,
    ),
)

print(sequences.split_counts())
```

Relevant code:
- [builder.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/sequences/builder.py)
- [artifacts.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/sequences/artifacts.py)

### 5. Baselines And THGKT Consume Those Views

Included models:
- logistic / MLP-style baseline
- DKT baseline
- graph-only baseline
- THGKT full model

The THGKT model has two latent state components:
- a graph-derived representation for structural educational context
- a temporal representation for the student's interaction trajectory

At a research level, the model can be read as:

1. Graph encoding:
   heterogeneous message passing produces latent embeddings
   `h_s`, `h_q`, and `h_c` for students, questions, and concepts

2. Temporal encoding:
   a sequence encoder maps the student's prefix history
   `x_1, ..., x_t`
   to a hidden knowledge-state vector
   `z_t`

3. Fusion and prediction:
   the model combines the target student embedding, target question embedding, target concept summary, and temporal state into a final logit

```text
h_G = GNN(G)
z_t = TemporalEncoder(x_1, ..., x_t)
u_t = Fuse(h_student, h_question, h_concepts, z_t, time_features)
P(y_{t+1}=1 | context) = sigmoid(MLP(u_t))
```

This gives a probabilistic interpretation of the model output: the final sigmoid is the estimated probability that the next response is correct under both the learner's recent trajectory and the current graph-defined educational context.

The temporal encoder can also be interpreted as an approximation to a latent student knowledge state vector, while the graph encoder supplies structural context that conditions that predictive distribution. In that sense, the model is not only classifying a response event; it is estimating correctness from a latent knowledge state informed by both temporal evidence and concept structure.

The graph encoder is implemented with `HeteroConv` and `SAGEConv`, the temporal encoder is a GRU, and the final head is a learned fusion network over student, question, concept, and temporal features.

Conceptual diagram:

```text
student history ----------> temporal encoder -----------.
                                                     |
heterogeneous graph -----> graph message passing ----+--> fusion --> prediction head --> P(correct_next=1)
                                                     |
target question / concepts --------------------------'
```

Relevant code:
- [baselines.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/models/baselines.py)
- [thgkt.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/models/thgkt.py)

The model interface looks like this at a high level:

```python
from thgkt.models.thgkt import THGKTConfig, THGKTModel

model = THGKTModel(
    THGKTConfig(
        num_students=100,
        num_questions=500,
        num_concepts=80,
        hidden_dim=32,
        temporal_hidden_dim=32,
        use_graph_encoder=True,
        use_temporal_encoder=True,
        use_time_features=True,
        use_prerequisite_edges=True,
        dropout=0.1,
    )
)
```

### 6. Training, Evaluation, And Reporting Are Config-Driven

The repository is designed to run from a config file.

A config controls:
- dataset path and dataset-specific options
- preprocessing and split parameters
- graph construction mode
- sequence limits
- model type and ablations
- training hyperparameters
- explainability toggles

Example config fragment:

```json
{
  "dataset": {
    "name": "ednet_path",
    "path": "data/raw/ednet/KT1",
    "contents_path": "data/raw/ednet/contents",
    "max_users": 20000,
    "relation_mode": "transition"
  },
  "sequence": {
    "max_history_length": 128
  },
  "model": {
    "type": "thgkt",
    "use_graph_encoder": true,
    "use_temporal_encoder": true,
    "use_prerequisite_edges": true
  }
}
```

Programmatic example:

```python
from thgkt.experiment import run_experiment_from_config

summary = run_experiment_from_config(
    "configs/ednet_20k/thgkt_full.json",
    output_root="artifacts/ednet_20k_suite",
)

print(summary["test_metrics"])
print(summary["artifacts"])
```

Relevant code:
- [experiment.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/experiment.py)
- [runner.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/training/runner.py)

## Benchmark Variants In This Repository

The EdNet-20k comparison suite runs these variants on the same split setup so the outputs are directly comparable:
- full THGKT
- DKT baseline on the same split
- graph-only model
- THGKT ablation with no prerequisite edges
- THGKT ablation with no temporal encoder
- THGKT with a different graph construction strategy using `cooccurrence` instead of `transition`

Config directory:
- [configs/ednet_20k](/d:/.Dev/.UNJOClasses/ML/Final%20Project/configs/ednet_20k)

## Why The EdNet-20k Setting Is Reasonable

The repository does not default to the full EdNet corpus because the current implementation materializes canonical bundles, graph artifacts, and sequence artifacts explicitly. That makes full-corpus experimentation expensive on a single workstation even when GPU training itself is available.

The `20,000`-student setting is therefore a research-practical benchmark rather than an arbitrary small sample:
- it is large enough that EdNet still yields a dataset with interactions on the order of millions rather than thousands
- it preserves the heterogeneous structure needed to test graph and temporal hypotheses on a nontrivial student population
- it allows repeated ablation studies on the same hardware budget, which is necessary for fair comparison across multiple models

Academically, the point of this setting is not to claim full-corpus saturation. It is to obtain a tractable benchmark that is still large enough for comparative evaluation while respecting memory constraints imposed by explicit graph and sequence artifact generation.

## Implementation And Usage

The sections below are the operational appendix for reproducing the experiments, regenerating plots, and running the benchmark suite from the command line.

## Main Commands

### Verify The Project

```powershell
python -m pytest -q
.\.python312\python.exe -m pytest -q
```

### Run The Default Phase 12 Smoke Suite

```powershell
.\.python312\python.exe run_project.py --device cuda
```

If you request `--device cuda` and CUDA is unavailable, `run_project.py` now warns and asks whether it should continue on CPU instead of silently falling back.

### Run One EdNet-20k THGKT Experiment

```powershell
.\.python312\python.exe run_project.py --device cuda --skip-tests --config configs/ednet_kt1.yaml --output-root artifacts/ednet_kt1_run --heartbeat-seconds 15
```

### Run The Full EdNet-20k Comparison Suite

```powershell
.\.python312\python.exe run_project.py --device cuda --skip-tests --output-root artifacts/ednet_20k_suite --config configs/ednet_20k/thgkt_full.json --config configs/ednet_20k/dkt_baseline.json --config configs/ednet_20k/graph_only.json --config configs/ednet_20k/thgkt_no_prereq.json --config configs/ednet_20k/thgkt_no_temporal.json --config configs/ednet_20k/thgkt_cooccurrence_graph.json --heartbeat-seconds 15
```

### Run A Single Config Directly

```powershell
.\.python312\python.exe scripts/run_experiment.py --config configs/phase12/thgkt_assistments.yaml --output-root artifacts/final_project_run
```

### Regenerate Plots For An Existing Run

```powershell
.\.python312\python.exe scripts/plot_run.py --run-dir artifacts/final_project_run/phase12_thgkt_assistments
```

### Compare Existing Run Directories

```powershell
.\.python312\python.exe scripts/compare_runs.py --run-dirs artifacts/final_project_run/phase12_baseline_assistments artifacts/final_project_run/phase12_thgkt_assistments artifacts/final_project_run/phase12_thgkt_ablation_assistments --metric auc --output artifacts/final_project_run/ablation_comparison.svg
```

### Run The Phase 12 Suite Script Directly

```powershell
.\.python312\python.exe scripts/run_phase12_suite.py --output-root artifacts/final_project_run
```

## Result Storage And Comparison Outputs

Every run writes a self-contained run directory with:
- canonical bundle
- split artifact
- graph artifact
- sequence artifact
- checkpoint
- metrics
- predictions
- plots
- config snapshots
- run summary

Multi-run `run_project.py` executions also create a shared comparison directory:
- `comparison/comparison_metrics.json`
- `comparison/comparison_metrics.csv`
- `comparison/comparison_report.md`
- `comparison/comparison_auc.svg`
- `comparison/comparison_accuracy.svg`
- `comparison/comparison_f1.svg`
- `comparison/comparison_bce_loss.svg`

This gives you both:
- machine-readable output for later analysis
- presentation-ready plots and a Markdown summary for reporting

## Repository Structure

- `src/thgkt/`: project package
- `configs/`: experiment configs
- `configs/phase12/`: original smoke-test configs
- `configs/ednet_20k/`: EdNet-20k benchmark configs
- `scripts/`: script entrypoints
- `tests/`: unit and integration tests
- `artifacts/`: generated experiment outputs
- `data/raw/ednet/`: extracted EdNet dataset files

## Environment Notes

This workspace uses a repo-local CUDA-capable Python interpreter:

```powershell
.\.python312\python.exe
```

Quick CUDA check:

```powershell
.\.python312\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Current Practical Dataset Setting

The full EdNet corpus is much larger than a convenient in-memory research run, so the benchmark configs in this repo currently use **20,000 students** as the practical comparison setting.

That is large enough to compare models meaningfully while still fitting the current pipeline design.

## Limitations

This repository is intended to be honest about its current research limits.

- The pipeline is still artifact-heavy and mostly in-memory, so full-corpus EdNet runs are constrained by RAM, disk I/O, and preprocessing time rather than only GPU throughput.
- Concept labels and concept-question mappings may contain educational noise inherited from the source datasets; the graph is only as reliable as those annotations.
- Prerequisite edges are generated or adapted from available concept relations and are therefore approximations of pedagogy rather than ground-truth cognitive structure.
- Sequence truncation with `max_history_length` improves tractability but may discard long-range history that matters for some learners.
- Graph sparsity and cold-start effects remain important for rare concepts, rare questions, and low-activity students.
- Because this is a course-project codebase, the emphasis is on reproducible comparison and ablation rather than exhaustive hyperparameter search or industrial-scale distributed training.

## Relevant Source Files

- [run_project.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/run_project.py)
- [experiment.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/experiment.py)
- [ednet.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/adapters/ednet.py)
- [preprocessing.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/data/preprocessing.py)
- [builder.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/graph/builder.py)
- [builder.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/sequences/builder.py)
- [baselines.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/models/baselines.py)
- [thgkt.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/models/thgkt.py)
- [runner.py](/d:/.Dev/.UNJOClasses/ML/Final%20Project/src/thgkt/training/runner.py)

## Known Notes

- `torch-geometric` emits an upstream deprecation warning in the current environment during tests.
- The plotting layer uses SVG generation directly, so it does not require `matplotlib`.
- The current pipeline is still artifact-heavy and in-memory relative to full-corpus EdNet scale.

