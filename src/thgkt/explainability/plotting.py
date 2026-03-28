"""Simple SVG plotting helpers for explainability outputs."""

from __future__ import annotations

from pathlib import Path


def save_bar_chart_svg(
    labels: list[str],
    values: list[float],
    path: str | Path,
    *,
    title: str,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 720
    height = 420
    margin_left = 140
    margin_top = 60
    plot_width = width - margin_left - 40
    bar_height = 32
    gap = 14
    max_value = max([abs(value) for value in values], default=1.0)
    max_value = max(max_value, 1e-6)

    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + index * (bar_height + gap)
        bar_width = int(plot_width * abs(value) / max_value)
        color = "#1f6feb" if value >= 0 else "#d73a49"
        bars.append(
            f'<text x="16" y="{y + 22}" font-size="14" font-family="Segoe UI">{_escape(label)}</text>'
            f'<rect x="{margin_left}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="4" />'
            f'<text x="{margin_left + bar_width + 10}" y="{y + 22}" font-size="13" font-family="Segoe UI">{value:.4f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="16" y="30" font-size="20" font-family="Segoe UI" font-weight="600">{_escape(title)}</text>
  <line x1="{margin_left}" y1="44" x2="{margin_left}" y2="{height - 24}" stroke="#999" stroke-width="1" />
  {''.join(bars)}
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
