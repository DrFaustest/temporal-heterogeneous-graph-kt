"""SVG plot generation for training and experiment reports."""

from __future__ import annotations

from pathlib import Path


def save_training_curves_svg(
    train_history: list[dict[str, float]],
    val_history: list[dict[str, float]],
    path: str | Path,
) -> Path:
    width = 760
    height = 420
    left = 60
    right = 30
    top = 40
    bottom = 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    epochs = [entry["epoch"] for entry in train_history]
    train_loss = [entry["loss"] for entry in train_history]
    val_bce = [entry["bce_loss"] for entry in val_history]
    y_values = train_loss + val_bce
    y_min = min(y_values)
    y_max = max(y_values)
    y_span = max(y_max - y_min, 1e-6)

    def point(epoch: float, value: float) -> tuple[float, float]:
        x = left + ((epoch - min(epochs)) / max(max(epochs) - min(epochs), 1.0)) * plot_w
        y = top + (1.0 - (value - y_min) / y_span) * plot_h
        return x, y

    train_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in [point(e, v) for e, v in zip(epochs, train_loss)])
    val_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in [point(e, v) for e, v in zip(epochs, val_bce)])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="20" y="26" font-size="20" font-family="Segoe UI" font-weight="600">Training Curves</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#666" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#666" />
  <polyline fill="none" stroke="#1f6feb" stroke-width="3" points="{train_points}" />
  <polyline fill="none" stroke="#d73a49" stroke-width="3" points="{val_points}" />
  <text x="{width - 210}" y="36" font-size="13" font-family="Segoe UI" fill="#1f6feb">Train Loss</text>
  <text x="{width - 120}" y="36" font-size="13" font-family="Segoe UI" fill="#d73a49">Val BCE</text>
</svg>
'''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def save_roc_curve_svg(
    probs: list[float],
    targets: list[int],
    path: str | Path,
) -> Path:
    roc_points = _roc_points(probs, targets)
    width = 420
    height = 420
    left = 50
    top = 30
    size = 320
    coords = []
    for fpr, tpr in roc_points:
        x = left + fpr * size
        y = top + (1.0 - tpr) * size
        coords.append(f"{x:.1f},{y:.1f}")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="20" y="22" font-size="20" font-family="Segoe UI" font-weight="600">ROC Curve</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + size}" stroke="#666" />
  <line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top + size}" stroke="#666" />
  <line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="#bbb" stroke-dasharray="6,6" />
  <polyline fill="none" stroke="#1f6feb" stroke-width="3" points="{' '.join(coords)}" />
</svg>
'''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def save_ablation_bar_chart_svg(
    labels: list[str],
    values: list[float],
    path: str | Path,
    *,
    title: str,
    metric_name: str,
) -> Path:
    width = 760
    height = 420
    left = 160
    top = 50
    bar_h = 30
    gap = 16
    plot_w = width - left - 40
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-6)
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * (bar_h + gap)
        bar_w = plot_w * value / max_value
        rows.append(
            f'<text x="16" y="{y + 20}" font-size="14" font-family="Segoe UI">{_escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="#1f6feb" rx="4" />'
            f'<text x="{left + bar_w + 8:.1f}" y="{y + 20}" font-size="13" font-family="Segoe UI">{value:.4f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="16" y="24" font-size="20" font-family="Segoe UI" font-weight="600">{_escape(title)}</text>
  <text x="16" y="42" font-size="13" font-family="Segoe UI" fill="#666">Metric: {_escape(metric_name)}</text>
  {''.join(rows)}
</svg>
'''
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _roc_points(probs: list[float], targets: list[int]) -> list[tuple[float, float]]:
    positives = sum(targets)
    negatives = len(targets) - positives
    if not probs:
        return [(0.0, 0.0), (1.0, 1.0)]

    ranked = sorted(zip(probs, targets), key=lambda item: item[0], reverse=True)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    tp = 0
    fp = 0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            _, target = ranked[index]
            if int(target) == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        points.append((fpr, tpr))

    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))
    return points


def _escape(value: str) -> str:
    return value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
