# Canonical Schema and Module Interface Specification

## Canonical Schema

The most important architectural rule is to force all datasets into a **shared canonical schema**.

## 1. Canonical Interaction Schema

### Required fields
- `student_id: str | int`
- `interaction_id: str | int`
- `seq_idx: int`
- `timestamp: datetime | int | float | None`
- `question_id: str | int`
- `correct: int`
- `source_dataset: str`

### Strongly recommended fields
- `concept_ids: list[str | int]`
- `elapsed_time: float | None`
- `attempt_count: int | None`
- `session_id: str | int | None`
- `response_time: float | None`

### Invariants
- Each row represents a single student-question interaction
- `seq_idx` must be valid within each student sequence
- `correct` must be normalized to binary
- `question_id` must exist
- `student_id` must exist
- `concept_ids` may be empty but must be present as a field

---

## 2. Canonical Question Table

### Fields
- `question_id`
- `question_text: str | None`
- `difficulty: float | None`
- `question_metadata: dict | None`

### Invariants
- `question_id` unique
- no duplicated question rows

---

## 3. Canonical Concept Table

### Fields
- `concept_id`
- `concept_name: str | None`
- `concept_metadata: dict | None`

### Invariants
- `concept_id` unique

---

## 4. Canonical Question-Concept Mapping Table

### Fields
- `question_id`
- `concept_id`
- `weight: float`
- `mapping_source: str`

### Invariants
- `weight > 0`
- `question_id` exists in question table
- `concept_id` exists in concept table

---

## 5. Canonical Concept Relation Table

### Fields
- `source_concept_id`
- `target_concept_id`
- `relation_type: str`
- `weight: float`
- `relation_source: str`

### Allowed relation types initially
- `prerequisite_of`
- `cooccurs_with`
- `transition_to`

### Invariants
- no self-loops unless explicitly allowed by config
- weights normalized if required by graph builder
- concepts must exist in concept table

---

## 6. Canonical Split Specification

### Split artifact fields
- `split_name`
- `student_ids` or interaction index groups
- `split_strategy`
- `timestamp`
- `random_seed`

### Rules
- no student overlap if using student-wise split
- within each student sequence, validation and test must occur later than training data
- if interaction-wise chronological splitting is used, document it clearly

## Module Interfaces

## 1. Dataset Adapter Interface

```python
class BaseDatasetAdapter(Protocol):
    def load_raw(self, raw_path: str | Path) -> Any: ...
    def to_canonical(self, raw_obj: Any) -> CanonicalDatasetBundle: ...
    def validate_canonical(self, bundle: CanonicalDatasetBundle) -> ValidationReport: ...
```

### Required implementations
- `AssistmentsAdapter`
- `EdNetAdapter`
- `SyntheticToyAdapter`

---

## 2. Canonical Dataset Bundle

```python
@dataclass
class CanonicalDatasetBundle:
    interactions: pd.DataFrame
    questions: pd.DataFrame
    concepts: pd.DataFrame
    question_concept_map: pd.DataFrame
    concept_relations: pd.DataFrame
    metadata: dict
```

---

## 3. Preprocessor Interface

```python
class DatasetPreprocessor(Protocol):
    def clean(self, bundle: CanonicalDatasetBundle) -> CanonicalDatasetBundle: ...
    def normalize(self, bundle: CanonicalDatasetBundle) -> CanonicalDatasetBundle: ...
    def filter(self, bundle: CanonicalDatasetBundle, config: dict) -> CanonicalDatasetBundle: ...
    def add_sequence_indices(self, bundle: CanonicalDatasetBundle) -> CanonicalDatasetBundle: ...
```

---

## 4. Splitter Interface

```python
class Splitter(Protocol):
    def make_splits(
        self,
        interactions: pd.DataFrame,
        strategy: str,
        config: dict
    ) -> SplitArtifacts: ...
```

### Supported strategies
- `student_chronological`
- `student_holdout`
- `interaction_chronological` only if explicitly enabled

### Default
`student_chronological`

---

## 5. Graph Builder Interface

```python
class GraphBuilder(Protocol):
    def build_hetero_graph(
        self,
        bundle: CanonicalDatasetBundle,
        split_artifacts: SplitArtifacts,
        config: dict
    ) -> HeteroGraphArtifacts: ...
```

---

## 6. Sequence Builder Interface

```python
class SequenceBuilder(Protocol):
    def build_student_sequences(
        self,
        interactions: pd.DataFrame,
        split_artifacts: SplitArtifacts,
        config: dict
    ) -> SequenceArtifacts: ...
```

---

## 7. Model Interface

```python
class BaseModel(nn.Module):
    def forward(self, batch: dict) -> dict:
        ...
```

### Standard forward output keys
- `logits`
- `probs`
- `targets`
- `aux_outputs` optional
- `debug_info` optional

### Required models
- `MLPBaseline`
- `DKTBaseline`
- `GraphOnlyModel`
- `THGKTModel`

---

## 8. Trainer Interface

```python
class Trainer(Protocol):
    def train_epoch(self, model, loader, optimizer) -> dict: ...
    def validate_epoch(self, model, loader) -> dict: ...
    def fit(self, model, train_loader, val_loader, config: dict) -> TrainingRunArtifacts: ...
```

---

## 9. Evaluator Interface

```python
class Evaluator(Protocol):
    def evaluate(self, model, loader) -> dict: ...
    def compare_runs(self, run_dirs: list[Path]) -> pd.DataFrame: ...
```

---

## 10. Explainability Interface

```python
class ExplainabilityEngine(Protocol):
    def concept_importance(self, model, batch: dict) -> dict: ...
    def prerequisite_influence(self, model, batch: dict) -> dict: ...
    def mastery_trajectory(self, model, student_id: str | int) -> dict: ...
```
