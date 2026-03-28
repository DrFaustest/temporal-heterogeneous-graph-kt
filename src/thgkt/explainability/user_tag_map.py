"""Per-user concept/tag map artifacts built from canonical bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import cos, pi, sin
from pathlib import Path
from typing import Any

from thgkt.data.io import save_json
from thgkt.schemas.canonical import CanonicalBundle


@dataclass(frozen=True, slots=True)
class UserTagMapConfig:
    edge_scope: str = "all_questions"
    min_edge_question_count: int = 1


@dataclass(frozen=True, slots=True)
class UserTagNode:
    tag: str
    correct_count: int
    incorrect_count: int
    total_attempts: int
    node_weight: int
    mastery_score: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "total_attempts": self.total_attempts,
            "node_weight": self.node_weight,
            "mastery_score": self.mastery_score,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True, slots=True)
class UserTagEdge:
    source_tag: str
    target_tag: str
    question_count: int
    edge_weight: float
    normalized_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tag": self.source_tag,
            "target_tag": self.target_tag,
            "question_count": self.question_count,
            "edge_weight": self.edge_weight,
            "normalized_weight": self.normalized_weight,
        }


@dataclass(frozen=True, slots=True)
class UserTagMapArtifacts:
    student_id: str
    nodes: tuple[UserTagNode, ...]
    edges: tuple[UserTagEdge, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


def build_user_tag_map(
    bundle: CanonicalBundle,
    student_id: str,
    config: UserTagMapConfig | None = None,
) -> UserTagMapArtifacts:
    map_config = config or UserTagMapConfig()
    if map_config.edge_scope not in {"all_questions", "seen_questions"}:
        raise ValueError(f"Unsupported edge scope: {map_config.edge_scope}")
    if map_config.min_edge_question_count < 1:
        raise ValueError("min_edge_question_count must be at least 1")

    question_to_tags = _question_to_tags(bundle)
    interactions = [
        dict(row)
        for row in bundle.interactions.rows
        if str(row["student_id"]) == str(student_id)
    ]
    interactions.sort(
        key=lambda row: (
            int(row.get("seq_idx", 0)),
            str(row.get("timestamp", "")),
            str(row.get("interaction_id", "")),
        )
    )
    if not interactions:
        raise ValueError(f"Student {student_id} was not found in the canonical bundle")

    node_counts: dict[str, dict[str, int]] = {}
    seen_question_ids: set[str] = set()
    for row in interactions:
        question_id = str(row["question_id"])
        seen_question_ids.add(question_id)
        tags = row.get("concept_ids") or question_to_tags.get(question_id, ())
        correct = int(row.get("correct", 0))
        for tag in sorted({str(tag) for tag in tags if str(tag).strip()}):
            stats = node_counts.setdefault(tag, {"correct": 0, "incorrect": 0})
            if correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

    user_tags = set(node_counts)
    if map_config.edge_scope == "all_questions":
        edge_question_ids = sorted(question_to_tags)
    else:
        edge_question_ids = sorted(seen_question_ids)

    edge_counts: dict[tuple[str, str], int] = {}
    for question_id in edge_question_ids:
        filtered_tags = sorted({tag for tag in question_to_tags.get(question_id, ()) if tag in user_tags})
        for source_tag, target_tag in combinations(filtered_tags, 2):
            edge_counts[(source_tag, target_tag)] = edge_counts.get((source_tag, target_tag), 0) + 1

    filtered_edge_counts = {
        pair: count
        for pair, count in edge_counts.items()
        if count >= map_config.min_edge_question_count
    }
    max_edge_count = max(filtered_edge_counts.values(), default=0)

    nodes = []
    for tag in sorted(node_counts):
        correct_count = int(node_counts[tag]["correct"])
        incorrect_count = int(node_counts[tag]["incorrect"])
        total_attempts = correct_count + incorrect_count
        accuracy = correct_count / total_attempts if total_attempts else 0.0
        node_weight = correct_count - incorrect_count
        mastery_score = node_weight / total_attempts if total_attempts else 0.0
        nodes.append(
            UserTagNode(
                tag=tag,
                correct_count=correct_count,
                incorrect_count=incorrect_count,
                total_attempts=total_attempts,
                node_weight=node_weight,
                mastery_score=mastery_score,
                accuracy=accuracy,
            )
        )

    edges = []
    for (source_tag, target_tag), question_count in sorted(filtered_edge_counts.items()):
        normalized_weight = question_count / max_edge_count if max_edge_count else 0.0
        edges.append(
            UserTagEdge(
                source_tag=source_tag,
                target_tag=target_tag,
                question_count=int(question_count),
                edge_weight=float(question_count),
                normalized_weight=float(normalized_weight),
            )
        )

    total_correct = sum(node.correct_count for node in nodes)
    total_incorrect = sum(node.incorrect_count for node in nodes)
    return UserTagMapArtifacts(
        student_id=str(student_id),
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "builder": "build_user_tag_map",
            "edge_scope": map_config.edge_scope,
            "min_edge_question_count": map_config.min_edge_question_count,
            "num_interactions": len(interactions),
            "num_seen_questions": len(seen_question_ids),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "total_correct_tag_hits": total_correct,
            "total_incorrect_tag_hits": total_incorrect,
            "max_edge_question_count": max_edge_count,
            "max_node_attempts": max((node.total_attempts for node in nodes), default=0),
        },
    )


def export_user_tag_map_artifacts(
    artifacts: UserTagMapArtifacts,
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "user_tag_map.json"
    svg_path = output_path / "user_tag_map.svg"
    save_json(artifacts.to_dict(), json_path)
    save_user_tag_map_svg(artifacts, svg_path)
    return {
        "json_path": str(json_path),
        "svg_path": str(svg_path),
    }


def save_user_tag_map_svg(artifacts: UserTagMapArtifacts, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1040
    height = 820
    center_x = width / 2
    center_y = height / 2 - 20
    radius = min(width, height) * 0.33
    max_attempts = max((node.total_attempts for node in artifacts.nodes), default=1)
    positions = _circular_positions([node.tag for node in artifacts.nodes], center_x, center_y, radius)

    edge_elements = []
    for edge in artifacts.edges:
        source_x, source_y = positions[edge.source_tag]
        target_x, target_y = positions[edge.target_tag]
        stroke_width = 1.5 + (5.5 * edge.normalized_weight)
        opacity = 0.25 + (0.55 * edge.normalized_weight)
        edge_elements.append(
            f'<line x1="{source_x:.2f}" y1="{source_y:.2f}" x2="{target_x:.2f}" y2="{target_y:.2f}" '
            f'stroke="#5b6b7a" stroke-width="{stroke_width:.2f}" stroke-opacity="{opacity:.2f}" />'
        )

    node_elements = []
    for node in artifacts.nodes:
        x, y = positions[node.tag]
        node_radius = 18.0 + (18.0 * node.total_attempts / max_attempts)
        fill = _mastery_color(node.mastery_score)
        node_elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{node_radius:.2f}" fill="{fill}" '
            'stroke="#17324d" stroke-width="2" />'
        )
        node_elements.append(
            f'<text x="{x:.2f}" y="{y - node_radius - 10:.2f}" text-anchor="middle" '
            'font-size="14" font-family="Segoe UI" fill="#102030">'
            f'{_escape(node.tag)}</text>'
        )
        node_elements.append(
            f'<text x="{x:.2f}" y="{y + 5:.2f}" text-anchor="middle" '
            'font-size="12" font-family="Segoe UI" font-weight="700" fill="#102030">'
            f'{node.correct_count}/{node.incorrect_count}</text>'
        )

    subtitle = (
        f"student={artifacts.student_id} | edge_scope={artifacts.metadata.get('edge_scope', 'unknown')} | "
        f"nodes={len(artifacts.nodes)} | edges={len(artifacts.edges)}"
    )
    legend = [
        "Node label: tag",
        "Inside node: correct/incorrect counts",
        "Node color: green=stronger history, red=weaker history",
        "Edge width: more question co-occurrences",
    ]
    legend_lines = []
    for index, line in enumerate(legend):
        legend_lines.append(
            f'<text x="36" y="{710 + (index * 24)}" font-size="14" font-family="Segoe UI" fill="#203040">'
            f'{_escape(line)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f4eb" />
  <rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="28" fill="#fffdf8" stroke="#d6d0c4" />
  <text x="36" y="56" font-size="28" font-family="Segoe UI" font-weight="700" fill="#102030">EdNet User Tag Map</text>
  <text x="36" y="86" font-size="15" font-family="Segoe UI" fill="#425466">{_escape(subtitle)}</text>
  {''.join(edge_elements)}
  {''.join(node_elements)}
  {''.join(legend_lines)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _question_to_tags(bundle: CanonicalBundle) -> dict[str, tuple[str, ...]]:
    question_to_tags: dict[str, set[str]] = {}
    for row in bundle.question_concept_map.rows:
        question_id = str(row["question_id"])
        question_to_tags.setdefault(question_id, set()).add(str(row["concept_id"]))
    return {
        question_id: tuple(sorted(tags))
        for question_id, tags in question_to_tags.items()
    }


def _circular_positions(tags: list[str], center_x: float, center_y: float, radius: float) -> dict[str, tuple[float, float]]:
    if not tags:
        return {}
    if len(tags) == 1:
        return {tags[0]: (center_x, center_y)}
    positions: dict[str, tuple[float, float]] = {}
    for index, tag in enumerate(tags):
        angle = (-pi / 2.0) + (2.0 * pi * index / len(tags))
        positions[tag] = (
            center_x + (radius * cos(angle)),
            center_y + (radius * sin(angle)),
        )
    return positions


def _mastery_color(mastery_score: float) -> str:
    clamped = max(-1.0, min(1.0, mastery_score))
    if clamped >= 0.0:
        red = int(216 - (84 * clamped))
        green = int(222 - (26 * clamped))
        blue = int(196 - (100 * clamped))
    else:
        magnitude = abs(clamped)
        red = int(228 - (10 * magnitude))
        green = int(210 - (92 * magnitude))
        blue = int(204 - (116 * magnitude))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
