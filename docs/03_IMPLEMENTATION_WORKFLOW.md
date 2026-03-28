# Implementation Workflow, Tests, and Completion Criteria

This document defines the locked build process. Do not skip phases.

## Phase 0 — Repository Scaffolding

### Objective
Create the research project structure and core configuration system.

### Tasks
1. Create repository folders
2. Add `pyproject.toml` or `requirements.txt`
3. Add `src/` package structure
4. Add `configs/`
5. Add `tests/`
6. Add basic README
7. Add lint/test commands if practical

### Deliverables
- clean repository layout
- importable package
- dependency file
- placeholder config files

### Tests
- package imports without error
- test runner executes
- config file can be loaded

### Completion criteria
Do not continue until:
- project installs locally
- `pytest` runs successfully with placeholder tests
- config parsing works

---

## Phase 1 — Schema Definitions and Validation Layer

### Objective
Create the canonical data structures first.

### Tasks
1. Define `CanonicalDatasetBundle`
2. Define validation utilities
3. Define schema checking functions
4. Implement reports for missing columns, invalid values, duplicate IDs, and sequence inconsistencies

### Deliverables
- canonical data classes
- schema validators
- validation report object

### Tests
1. Valid synthetic bundle passes
2. Missing required columns fail with clear messages
3. Invalid `correct` values are caught
4. Duplicate IDs in concept table are caught
5. Empty concept lists are allowed if documented

### Completion criteria
Do not continue until:
- all schema validation tests pass
- error messages are readable and specific
- canonical bundle is the single required output type for all adapters

---

## Phase 2 — Synthetic Toy Dataset Adapter

### Objective
Build a toy dataset path first so the entire pipeline can be smoke-tested without external data.

### Tasks
1. Implement `SyntheticToyAdapter`
2. Generate:
   - a few students
   - a few questions
   - a few concepts
   - deterministic interaction sequences
   - known concept mappings
3. Produce canonical bundle

### Deliverables
- synthetic data generator
- canonical output
- known expected statistics

### Tests
1. bundle validates
2. expected number of students/questions/concepts match
3. sequence order is correct
4. graph can later be built from this without special casing

### Completion criteria
Do not continue until:
- synthetic bundle passes schema validation
- toy data is sufficient to test full end-to-end pipeline

---

## Phase 3 — ASSISTments Dataset Adapter

### Objective
Implement the primary real dataset adapter.

### Tasks
1. Build raw data loader
2. Map source columns to canonical columns
3. Normalize correctness labels
4. Extract or derive concept mappings
5. Add sequence order
6. Generate metadata report

### Deliverables
- `AssistmentsAdapter`
- canonicalized ASSISTments dataset
- dataset report describing counts and missingness

### Tests
1. raw file loads
2. canonical bundle validates
3. student sequences are monotonic
4. question-concept mappings exist where expected
5. adapter does not hardcode one exact filename or brittle column names if avoidable

### Completion criteria
Do not continue until:
- a small ASSISTments subset can be loaded end-to-end
- the canonical bundle is saved to disk
- summary statistics are printed or saved

---

## Phase 4 — Dataset Cleaning, Filtering, and Split Logic

### Objective
Produce reliable train/validation/test splits without leakage.

### Tasks
1. Implement preprocessing:
   - drop broken rows
   - normalize nulls
   - filter ultra-short sequences
   - normalize concept fields
2. Implement split creation
3. Save split artifacts

### Tests
1. no overlap between train/val/test students if student-wise split
2. for chronological split, validation and test are later than training within each student
3. split artifacts can be reloaded
4. running with same seed yields same split

### Completion criteria
Do not continue until:
- split reproducibility is verified
- leakage checks pass
- split artifacts are persisted

---

## Phase 5 — Graph Construction Layer

### Objective
Build the heterogeneous graph from canonical data.

### Tasks
1. Create node mappings:
   - student index map
   - question index map
   - concept index map
2. Build edge sets:
   - student answered question
   - question tests concept
   - concept prerequisite_of concept
   - optional reverse edges if configured
3. Create node features
4. Package graph as PyTorch Geometric-compatible artifact

### Tests
1. node counts match unique IDs
2. edge counts match source tables
3. no invalid indices
4. graph object serializes
5. relation metadata is correct

### Completion criteria
Do not continue until:
- hetero graph builds successfully from toy data and ASSISTments subset
- graph summary report matches input statistics
- graph artifacts save and reload correctly

---

## Phase 6 — Sequence Construction Layer

### Objective
Build sequential next-response training examples.

### Tasks
1. group interactions by student
2. sort by `seq_idx`
3. create prefix-to-next-response training tuples
4. encode question/concept/correctness/time features
5. support padding or packed sequences

### Tests
1. each target truly represents the next interaction
2. sequence masks are correct
3. no future leakage in prefixes
4. batch collation works for variable lengths

### Completion criteria
Do not continue until:
- sequence builder works on toy and ASSISTments data
- sample batches inspect correctly
- next-response tuples are verifiably correct

---

## Phase 7 — Baseline Models

### Objective
Create baselines before the full hybrid model.

### Tasks
1. Implement `MLPBaseline`
2. Implement `DKTBaseline`
3. Implement `GraphOnlyModel`

### Design requirement
All models must use the same training and evaluation interface.

### Tests
1. forward pass works
2. loss computes
3. one training step decreases or at least computes valid gradients
4. outputs contain required keys
5. shapes match target batch shapes

### Completion criteria
Do not continue until:
- all baselines train for at least one mini-run
- evaluation pipeline works on baseline outputs
- metrics are logged successfully

---

## Phase 8 — Training and Evaluation Framework

### Objective
Create robust experiment execution before the main model.

### Tasks
1. implement trainer
2. implement evaluator
3. add checkpointing
4. add early stopping
5. add run directory structure
6. add metric logging
7. add config-driven execution

### Tests
1. train loop runs for 1–3 epochs on toy data
2. checkpoints save and reload
3. evaluator computes all required metrics
4. early stopping triggers correctly
5. bad config produces readable failure

### Completion criteria
Do not continue until:
- complete train/eval cycle works for at least one baseline
- metrics persist to file
- checkpoint reload reproduces evaluation

---

## Phase 9 — Main Hybrid Model: THGKT

### Objective
Build the full temporal heterogeneous graph model only after all prior infrastructure is working.

### Proposed architecture
1. Node embeddings
2. Heterogeneous graph encoder
3. Temporal encoder over student history
4. Fusion layer
5. Prediction head

### Tasks
1. implement relation-aware heterogeneous graph encoder
2. implement temporal encoder
3. define fusion between graph context and temporal state
4. implement prediction head
5. support ablation flags

### Ablation flags required
- disable graph encoder
- disable temporal encoder
- remove prerequisite edges
- remove time features
- collapse heterogeneous relation typing if feasible

### Tests
1. forward pass works on real batch
2. graph encoder output dimensions align with temporal encoder inputs
3. ablation flags do not break batch processing
4. logits and targets align
5. gradients propagate through all enabled modules

### Completion criteria
Do not continue until:
- full model trains on toy data
- full model trains on small ASSISTments subset
- ablation toggles work without code changes elsewhere

---

## Phase 10 — Concept Relation Construction Variants

### Objective
Enable the ambitious part: multiple ways to build concept-concept edges.

### Tasks
1. implement empty/no-prerequisite option
2. implement co-occurrence graph option
3. implement transition-based graph option
4. allow provided graph if dataset has one
5. persist relation-generation metadata

### Tests
1. each graph mode produces valid concept relation table
2. no-prerequisite mode still allows training
3. co-occurrence weights compute deterministically
4. transition-based relations respect order statistics
5. changing graph mode changes artifact fingerprint

### Completion criteria
Do not continue until:
- at least two relation-generation strategies run successfully
- experiment configs can switch graph construction without code edits

---

## Phase 11 — Explainability and Analysis Outputs

### Objective
Add interpretation utilities after the main model is stable.

### Tasks
1. concept importance summaries
2. prerequisite edge influence summaries
3. mastery trajectory plotting
4. example-case export for paper figures

### Tests
1. explainability module runs on trained checkpoint
2. outputs are saved without crashing
3. at least one interpretable artifact is human-readable
4. plotting works from saved run directory

### Completion criteria
Do not continue until:
- one trained model produces usable explanation artifacts
- visual outputs are ready for inclusion in paper figures

---

## Phase 12 — Plotting, Experiment Scripts, and Final Polish

### Objective
Make the project publication-ready and reproducible.

### Tasks
1. add training-curve plots
2. add ROC curve plots
3. add ablation bar charts
4. add scripts for common runs
5. expand README with exact reproduction steps
6. add experiment matrix templates

### Tests
1. plots generate from saved logs
2. scripts launch correct config
3. README instructions are sufficient for fresh setup
4. final smoke test from clean environment passes

### Completion criteria
Project is considered complete only when:
- one full baseline run completes from raw data to metrics
- one full THGKT run completes from raw data to metrics
- at least one ablation run completes
- saved outputs include logs, config, checkpoint, metrics, and plots
- README explains end-to-end execution

## Locked Progression Rules

1. Do not implement the main model before:
   - canonical schema exists
   - dataset adapter works
   - splits are validated
   - graph and sequence builders both work

2. Do not implement EdNet support before:
   - ASSISTments pipeline works end-to-end
   - toy dataset tests pass
   - baseline training works

3. Do not add explainability features before:
   - full model trains
   - evaluation metrics are computed
   - checkpoints can be loaded reliably

4. Do not merge dataset-specific assumptions into model code

5. Every major artifact must be serializable:
   - canonical bundles
   - split artifacts
   - graph artifacts
   - sequence artifacts
   - configs
   - metrics
   - checkpoints

## Testing Matrix

### Unit Tests
- schema validation
- adapter normalization
- split logic
- graph construction
- sequence construction
- model forward pass
- metric computation

### Integration Tests
- toy dataset end-to-end
- ASSISTments subset end-to-end
- baseline train/eval cycle
- full model train/eval cycle

### Regression Tests
- same seed -> same split
- same saved checkpoint -> same evaluation metrics within tolerance
- graph artifact reload remains valid

### Failure Handling Tests
- missing columns
- malformed concept mappings
- empty student sequence after filtering
- invalid config
- unsupported graph mode

## Entire Project Completion Criteria

### Data pipeline completion
- toy dataset adapter works
- ASSISTments adapter works
- canonical schema validation passes
- split artifacts are reproducible
- graph artifacts build correctly
- sequence artifacts build correctly

### Modeling completion
- MLP baseline trains
- DKT baseline trains
- graph-only model trains
- THGKT model trains

### Evaluation completion
- AUC, Accuracy, F1, and BCE are computed
- results are saved to disk
- at least one comparison table can be generated

### Research completion
- at least one ablation is successful
- at least one graph construction variant is compared
- at least one explainability artifact is generated

### Reproducibility completion
- configs drive experiments
- saved run directory contains all needed artifacts
- README documents setup and run commands
