"""ASSISTments-style CSV adapter."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from thgkt.data.bundles import make_canonical_bundle
from thgkt.data.summaries import summarize_bundle
from thgkt.schemas.canonical import CanonicalBundle
from thgkt.schemas.validators import ValidationReport, validate_bundle


@dataclass(frozen=True, slots=True)
class AssistmentsAdapterConfig:
    source_dataset: str = "assistments"


class AssistmentsAdapter:
    """Load ASSISTments-style flat files into the canonical schema."""

    _COLUMN_ALIASES = {
        "student_id": ("student_id", "user_id", "user", "student", "studentid"),
        "question_id": ("question_id", "problem_id", "item_id", "problem", "question"),
        "correct": ("correct", "is_correct", "answered_correctly", "label"),
        "timestamp": ("timestamp", "start_time", "timestamp_ms", "time"),
        "elapsed_time": ("elapsed_time", "ms_first_response", "response_time", "duration"),
        "attempt_count": ("attempt_count", "attempts", "attempt", "attempt_number"),
        "interaction_id": ("interaction_id", "log_id", "row_id", "event_id"),
        "seq_idx": ("seq_idx", "sequence_id", "order_id"),
        "concept_ids": (
            "concept_ids",
            "concept_id",
            "skill_id",
            "skill_ids",
            "skill",
            "kc_id",
            "kc",
        ),
    }

    def __init__(self, config: AssistmentsAdapterConfig | None = None) -> None:
        self.config = config or AssistmentsAdapterConfig()

    def load_raw(self, raw_path: str | Path) -> dict[str, Any]:
        path = Path(raw_path)
        file_path = self._resolve_input_file(path)
        delimiter = "\t" if file_path.suffix.lower() in {".tsv", ".txt"} else ","
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = [dict(row) for row in reader]

        if not rows:
            raise ValueError(f"No rows found in ASSISTments file: {file_path}")

        return {"rows": rows, "path": str(file_path)}

    def to_canonical(self, raw_obj: Any) -> CanonicalBundle:
        rows = list(raw_obj["rows"])
        source_path = str(raw_obj.get("path", "unknown"))
        interactions = self._build_interactions(rows)
        questions = self._build_questions(interactions)
        concepts = self._build_concepts(interactions)
        question_concept_map = self._build_question_concept_map(interactions)

        bundle = make_canonical_bundle(
            interactions=interactions,
            questions=questions,
            concepts=concepts,
            question_concept_map=question_concept_map,
            concept_relations=[],
            metadata={
                "dataset_name": Path(source_path).stem,
                "source_dataset": self.config.source_dataset,
                "adapter": "AssistmentsAdapter",
                "raw_path": source_path,
            },
        )
        summary = summarize_bundle(bundle)
        bundle.metadata["summary"] = summary.to_dict()
        return bundle

    def validate_canonical(self, bundle: CanonicalBundle) -> ValidationReport:
        return validate_bundle(bundle)

    def _resolve_input_file(self, path: Path) -> Path:
        if path.is_file():
            return path
        if not path.exists():
            raise FileNotFoundError(f"ASSISTments path does not exist: {path}")
        candidates = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".csv", ".tsv", ".txt"}
        )
        if not candidates:
            raise FileNotFoundError(f"No CSV/TSV/TXT data files found under: {path}")
        return candidates[0]

    def _build_interactions(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        by_student: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row_idx, raw_row in enumerate(rows):
            student_id = self._required(raw_row, "student_id")
            question_id = self._required(raw_row, "question_id")
            correct = self._normalize_correct(self._required(raw_row, "correct"))
            timestamp_value = self._optional(raw_row, "timestamp") or str(row_idx)
            interaction_id = self._optional(raw_row, "interaction_id") or f"{student_id}-{row_idx}"
            elapsed_time = self._normalize_float(self._optional(raw_row, "elapsed_time"), default=0.0)
            attempt_count = self._normalize_int(self._optional(raw_row, "attempt_count"), default=1)
            seq_idx = self._optional(raw_row, "seq_idx")
            concept_ids = self._normalize_concepts(self._optional(raw_row, "concept_ids"))

            normalized = {
                "student_id": str(student_id),
                "interaction_id": str(interaction_id),
                "seq_idx": -1 if seq_idx in (None, "") else self._normalize_int(seq_idx, default=-1),
                "timestamp": str(timestamp_value),
                "question_id": str(question_id),
                "correct": correct,
                "concept_ids": concept_ids,
                "elapsed_time": elapsed_time,
                "attempt_count": attempt_count,
                "source_dataset": self.config.source_dataset,
            }
            normalized_rows.append(normalized)
            by_student[normalized["student_id"]].append(normalized)

        ordered_rows: list[dict[str, Any]] = []
        for student_id, student_rows in by_student.items():
            def _sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
                raw_seq = int(item["seq_idx"])
                effective_seq = raw_seq if raw_seq >= 0 else 10**9
                return (effective_seq, str(item["timestamp"]), str(item["interaction_id"]))

            student_rows_sorted = sorted(student_rows, key=_sort_key)
            for seq_idx, row in enumerate(student_rows_sorted):
                row["seq_idx"] = seq_idx
                ordered_rows.append(row)

        ordered_rows.sort(key=lambda row: (str(row["student_id"]), int(row["seq_idx"])))
        return ordered_rows

    def _build_questions(self, interactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        question_ids = sorted({str(row["question_id"]) for row in interactions})
        return [
            {"question_id": question_id, "source_dataset": self.config.source_dataset}
            for question_id in question_ids
        ]

    def _build_concepts(self, interactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        concept_ids = sorted({concept for row in interactions for concept in row["concept_ids"]})
        return [
            {"concept_id": concept_id, "source_dataset": self.config.source_dataset}
            for concept_id in concept_ids
        ]

    def _build_question_concept_map(
        self, interactions: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        mapping_pairs = {
            (str(row["question_id"]), str(concept_id))
            for row in interactions
            for concept_id in row["concept_ids"]
        }
        return [
            {"question_id": question_id, "concept_id": concept_id}
            for question_id, concept_id in sorted(mapping_pairs)
        ]

    def _find_key(self, row: dict[str, str], logical_name: str) -> str | None:
        aliases = self._COLUMN_ALIASES[logical_name]
        lowered = {key.lower(): key for key in row.keys()}
        for alias in aliases:
            if alias.lower() in lowered:
                return lowered[alias.lower()]
        return None

    def _required(self, row: dict[str, str], logical_name: str) -> str:
        key = self._find_key(row, logical_name)
        if key is None:
            raise KeyError(
                f"ASSISTments adapter could not find a column for '{logical_name}'. "
                f"Accepted aliases: {self._COLUMN_ALIASES[logical_name]}"
            )
        value = row[key]
        if value is None or str(value).strip() == "":
            raise ValueError(f"Column '{key}' is empty for required field '{logical_name}'")
        return str(value).strip()

    def _optional(self, row: dict[str, str], logical_name: str) -> str | None:
        key = self._find_key(row, logical_name)
        if key is None:
            return None
        value = row[key]
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped if stripped else None

    def _normalize_correct(self, value: str) -> int:
        lowered = value.strip().lower()
        truthy = {"1", "true", "t", "yes", "correct"}
        falsy = {"0", "false", "f", "no", "incorrect"}
        if lowered in truthy:
            return 1
        if lowered in falsy:
            return 0
        raise ValueError(f"Cannot normalize correctness label: {value!r}")

    def _normalize_float(self, value: str | None, *, default: float) -> float:
        if value is None:
            return default
        return float(value)

    def _normalize_int(self, value: str | None, *, default: int) -> int:
        if value is None:
            return default
        return int(float(value))

    def _normalize_concepts(self, value: str | None) -> list[str]:
        if value is None:
            return []
        normalized = value
        for delimiter in ("~~", ";", "|"):
            normalized = normalized.replace(delimiter, ",")
        concept_ids = [token.strip() for token in normalized.split(",") if token.strip()]
        return sorted(set(concept_ids))
