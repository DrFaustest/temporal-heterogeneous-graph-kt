"""EdNet KT1 dataset adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thgkt.data.bundles import make_canonical_bundle
from thgkt.data.summaries import summarize_bundle
from thgkt.schemas.canonical import CanonicalBundle
from thgkt.schemas.validators import ValidationReport, validate_bundle


@dataclass(frozen=True, slots=True)
class EdNetAdapterConfig:
    source_dataset: str = "ednet"
    max_users: int = 0
    user_ids: tuple[str, ...] = ()


class EdNetAdapter:
    """Load EdNet KT1 plus question metadata into the canonical schema."""

    def __init__(self, config: EdNetAdapterConfig | None = None) -> None:
        self.config = config or EdNetAdapterConfig()

    def load_raw(self, raw_path: str | Path, contents_path: str | Path | None = None) -> dict[str, Any]:
        kt1_dir = self._resolve_kt1_dir(Path(raw_path))
        questions_path = self._resolve_questions_file(Path(contents_path) if contents_path else kt1_dir.parent)
        user_files = sorted(kt1_dir.glob('*.csv'))
        if self.config.user_ids:
            requested_user_ids = {str(user_id).strip() for user_id in self.config.user_ids if str(user_id).strip()}
            user_files = [path for path in user_files if path.stem in requested_user_ids]
            loaded_user_ids = {path.stem for path in user_files}
            missing_user_ids = sorted(requested_user_ids - loaded_user_ids)
            if missing_user_ids:
                missing_display = ', '.join(missing_user_ids)
                raise FileNotFoundError(
                    f"Could not locate EdNet user files for: {missing_display} under {kt1_dir}"
                )
        if self.config.max_users > 0:
            user_files = user_files[: self.config.max_users]
        if not user_files:
            raise FileNotFoundError(f"No EdNet user files found under: {kt1_dir}")
        questions = self._load_questions(questions_path)
        return {
            "kt1_dir": str(kt1_dir),
            "questions_path": str(questions_path),
            "questions": questions,
            "user_files": [str(path) for path in user_files],
        }

    def to_canonical(self, raw_obj: Any) -> CanonicalBundle:
        questions_meta = raw_obj["questions"]
        interactions: list[dict[str, Any]] = []
        question_table: list[dict[str, Any]] = []
        concept_ids: set[str] = set()
        question_concept_pairs: set[tuple[str, str]] = set()

        for question_id in sorted(questions_meta):
            meta = questions_meta[question_id]
            question_table.append({"question_id": question_id, "source_dataset": self.config.source_dataset})
            for concept_id in meta["concept_ids"]:
                concept_ids.add(concept_id)
                question_concept_pairs.add((question_id, concept_id))

        for user_file_str in raw_obj["user_files"]:
            user_file = Path(user_file_str)
            student_id = user_file.stem
            with user_file.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                rows = [dict(row) for row in reader]
            rows.sort(key=lambda row: (int(row['timestamp']), int(row.get('solving_id', 0)), str(row['question_id'])))
            for seq_idx, row in enumerate(rows):
                question_id = str(row['question_id']).strip()
                if question_id not in questions_meta:
                    continue
                meta = questions_meta[question_id]
                user_answer = str(row['user_answer']).strip().lower()
                correct_answer = str(meta['correct_answer']).strip().lower()
                interactions.append(
                    {
                        "student_id": student_id,
                        "interaction_id": f"{student_id}-{seq_idx}",
                        "seq_idx": seq_idx,
                        "timestamp": str(row['timestamp']).strip(),
                        "question_id": question_id,
                        "correct": int(user_answer == correct_answer),
                        "concept_ids": list(meta['concept_ids']),
                        "elapsed_time": float(row.get('elapsed_time', 0) or 0),
                        "attempt_count": 1,
                        "source_dataset": self.config.source_dataset,
                    }
                )

        concepts = [
            {"concept_id": concept_id, "source_dataset": self.config.source_dataset}
            for concept_id in sorted(concept_ids)
        ]
        question_concept_map = [
            {"question_id": question_id, "concept_id": concept_id}
            for question_id, concept_id in sorted(question_concept_pairs)
        ]
        bundle = make_canonical_bundle(
            interactions=interactions,
            questions=question_table,
            concepts=concepts,
            question_concept_map=question_concept_map,
            concept_relations=[],
            metadata={
                "dataset_name": Path(raw_obj['kt1_dir']).name,
                "source_dataset": self.config.source_dataset,
                "adapter": "EdNetAdapter",
                "raw_path": raw_obj['kt1_dir'],
                "questions_path": raw_obj['questions_path'],
                "num_user_files_loaded": len(raw_obj['user_files']),
            },
        )
        summary = summarize_bundle(bundle)
        bundle.metadata['summary'] = summary.to_dict()
        return bundle

    def validate_canonical(self, bundle: CanonicalBundle) -> ValidationReport:
        return validate_bundle(bundle)

    def _resolve_kt1_dir(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Could not resolve EdNet KT1 directory from: {path}")
        if path.is_dir() and any(path.glob('u*.csv')):
            return path
        if path.is_dir() and any(path.glob('*.csv')):
            return path
        if path.is_dir() and (path / 'KT1').exists():
            return self._resolve_kt1_dir(path / 'KT1')
        if path.is_dir():
            candidates = [candidate for candidate in path.rglob('*') if candidate.is_dir() and any(candidate.glob('u*.csv'))]
            if candidates:
                return sorted(candidates)[0]
            csv_candidates = [candidate for candidate in path.rglob('*') if candidate.is_dir() and any(candidate.glob('*.csv'))]
            if csv_candidates:
                return sorted(csv_candidates)[0]
        raise FileNotFoundError(f"Could not resolve EdNet KT1 directory from: {path}")

    def _resolve_questions_file(self, path: Path) -> Path:
        candidates = []
        if path.is_file() and path.name.lower() == 'questions.csv':
            return path
        if path.is_dir():
            candidates.extend(path.rglob('questions.csv'))
        if not candidates:
            raise FileNotFoundError(f"Could not locate EdNet questions.csv under: {path}")
        return sorted(candidates)[0]

    def _load_questions(self, path: Path) -> dict[str, dict[str, Any]]:
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
        questions: dict[str, dict[str, Any]] = {}
        for row in rows:
            question_id = str(row['question_id']).strip()
            tags_value = str(row.get('tags', '')).strip()
            concept_ids = [token.strip() for token in tags_value.replace('|', ';').replace(',', ';').split(';') if token.strip()]
            if not concept_ids:
                part_value = str(row.get('part', '')).strip()
                if part_value:
                    concept_ids = [part_value]
            if not concept_ids:
                concept_ids = [f"question::{question_id}"]
            questions[question_id] = {
                "correct_answer": str(row.get('correct_answer', '')).strip(),
                "concept_ids": sorted(set(concept_ids)),
            }
        return questions


