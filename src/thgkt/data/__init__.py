"""Data adapters, preprocessing, splitting, relation generation, and artifact IO."""

from thgkt.data.adapters import AssistmentsAdapter, EdNetAdapter, SyntheticToyAdapter
from thgkt.data.artifacts import DatasetSummary, SplitArtifacts, SplitIndices
from thgkt.data.bundles import make_canonical_bundle
from thgkt.data.io import (
    load_canonical_bundle,
    load_json,
    load_split_artifacts,
    save_canonical_bundle,
    save_json,
    save_split_artifacts,
)
from thgkt.data.preprocessing import CanonicalPreprocessor, PreprocessingConfig
from thgkt.data.relations import RelationConfig, apply_concept_relation_mode, generate_concept_relations
from thgkt.data.splitting import (
    SplitConfig,
    check_chronological_no_leakage,
    check_no_student_overlap,
    make_splits,
)
from thgkt.data.summaries import summarize_bundle

__all__ = [
    "AssistmentsAdapter",
    "CanonicalPreprocessor",
    "DatasetSummary",
    "EdNetAdapter",
    "PreprocessingConfig",
    "RelationConfig",
    "SplitArtifacts",
    "SplitConfig",
    "SplitIndices",
    "SyntheticToyAdapter",
    "apply_concept_relation_mode",
    "check_chronological_no_leakage",
    "check_no_student_overlap",
    "generate_concept_relations",
    "load_canonical_bundle",
    "load_json",
    "load_split_artifacts",
    "make_canonical_bundle",
    "make_splits",
    "save_canonical_bundle",
    "save_json",
    "save_split_artifacts",
    "summarize_bundle",
]
