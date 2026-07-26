"""Rendering for :class:`~warrant.analysis.report.AuditReport`.

Produces a single self-contained HTML page — no external assets, theme-aware
(light/dark), responsive — so the audit is shareable as one file. Kept separate
from the analysis layer so ``audit()`` returns a usable data object even where
rendering deps are absent; the report model imports this lazily.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from warrant.analysis.report import AuditReport, NodeFinding

_VERDICT_COLOR = {"KEEP": "#2f9e44", "COLLAPSE": "#e8590c", "REVIEW": "#f08c00"}


def _bar_svg(findings: "list[NodeFinding]", max_dollars: float) -> str:
    """Inline horizontal bar chart of $/mo per node, coloured by verdict."""
    if not findings:
        return ""
    row_h, gap, label_w, chart_w = 26, 10, 130, 320
    height = len(findings) * (row_h + gap) + gap
    rows: list[str] = []
    for i, f in enumerate(findings):
        y = gap + i * (row_h + gap)
        width = 0 if max_dollars <= 0 else max(2, f.dollars_per_month / max_dollars * chart_w)
        color = _VERDICT_COLOR.get(f.recommendation.value, "#868e96")
        name = html.escape(f.node_id)
        rows.append(
            f'<text x="{label_w - 8}" y="{y + row_h * 0.68}" text-anchor="end" '
            f'class="lbl">{name}</text>'
            f'<rect x="{label_w}" y="{y}" width="{width:.1f}" height="{row_h}" '
            f'rx="3" fill="{color}"><title>{name}: ${f.dollars_per_month:,.2f}/mo</title></rect>'
            f'<text x="{label_w + width + 6:.1f}" y="{y + row_h * 0.68}" '
            f'class="val">${f.dollars_per_month:,.0f}/mo</text>'
        )
    return (
        f'<svg viewBox="0 0 {label_w + chart_w + 90} {height}" '
        f'width="100%" role="img" aria-label="Monthly cost per node">'
        + "".join(rows)
        + "</svg>"
    )


def _finding_row(f: "NodeFinding") -> str:
    color = _VERDICT_COLOR.get(f.recommendation.value, "#868e96")
    abl = "—" if f.ablation_value is None else f"{f.ablation_value:.2f}"
    nov = "—" if f.mean_novelty is None else f"{f.mean_novelty:.2f}"
    return (
        "<tr>"
        f'<td class="mono">{html.escape(f.node_id)}</td>'
        f'<td><span class="badge" style="background:{color}">{f.recommendation.value}</span></td>'
        f"<td>{f.admissibility.value}</td>"
        f'<td class="num">{abl}</td>'
        f'<td class="num">{nov}</td>'
        f'<td class="num">${f.dollars_per_month:,.2f}</td>'
        f'<td class="num">{f.confidence:.0%}</td>'
        f'<td class="reason">{html.escape(f.reason)}</td>'
        "</tr>"
    )


def render_html(report: "AuditReport") -> str:
    max_dollars = max((f.dollars_per_month for f in report.findings), default=0.0)
    rows = "".join(_finding_row(f) for f in report.findings)
    chart = _bar_svg(report.findings, max_dollars)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in report.notes)
    collapsible = report.collapsible()
    savings_line = (
        f"Collapsing {len(collapsible)} redundant agent(s) saves "
        f"<strong>${report.projected_savings_per_month:,.2f}/mo</strong> "
        f"of ${report.total_dollars_per_month:,.2f}/mo total."
    )
    distortion = (
        f'<p class="sub">Mean posterior chain-loss: '
        f"{report.mean_chain_loss_bits:.3f} bits across annotated decision chains.</p>"
        if report.distortion_available
        else ""
    )
    return _PAGE.format(
        graph=html.escape(report.graph_name or "graph"),
        n_runs=report.n_runs,
        savings_line=savings_line,
        chart=chart,
        rows=rows,
        notes=f"<ul class='notes'>{notes}</ul>" if notes else "",
        distortion=distortion,
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Warrant delegation audit — {graph}</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#1a1c1e; --muted:#6b7280; --line:#e5e7eb; --card:#f8f9fa;
    --accent:#1c7ed6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#121417; --fg:#e6e8ea; --muted:#9aa0a6; --line:#2a2e33; --card:#1a1d21; }}
  }}
  :root[data-theme="dark"] {{ --bg:#121417; --fg:#e6e8ea; --muted:#9aa0a6; --line:#2a2e33; --card:#1a1d21; }}
  :root[data-theme="light"] {{ --bg:#ffffff; --fg:#1a1c1e; --muted:#6b7280; --line:#e5e7eb; --card:#f8f9fa; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); margin:.15rem 0; }}
  .headline {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:1rem 1.25rem; margin:1.5rem 0; font-size:1.05rem; }}
  .chart {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:1rem; margin:1.5rem 0; overflow-x:auto; }}
  .lbl {{ fill:var(--fg); font-size:12px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .val {{ fill:var(--muted); font-size:11px; }}
  .tablewrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th, td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
    vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
    letter-spacing:.03em; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  td.reason {{ color:var(--muted); font-size:12.5px; min-width:220px; }}
  .badge {{ display:inline-block; color:#fff; padding:.1rem .5rem; border-radius:999px;
    font-size:11px; font-weight:700; letter-spacing:.02em; }}
  .notes {{ color:var(--muted); font-size:13px; padding-left:1.1rem; }}
  footer {{ margin-top:2.5rem; color:var(--muted); font-size:12px; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Warrant — delegation audit</h1>
  <p class="sub">Graph <strong>{graph}</strong> · {n_runs} run(s) observed</p>
  <div class="headline">{savings_line}</div>
  {distortion}
  <div class="chart">{chart}</div>
  <div class="tablewrap">
  <table>
    <thead><tr>
      <th>Node</th><th>Verdict</th><th>Class</th><th>Ablation</th><th>Novelty</th>
      <th>$/mo</th><th>Conf.</th><th>Reason</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  {notes}
  <footer>
    Ablation = fraction of runs whose final answer changed when the node was disabled
    (0 ⇒ collapsible, 1 ⇒ load-bearing). Novelty = fraction of a node's output tokens
    absent from prior context. Grounded in Ao, Gao &amp; Simchi-Levi, arXiv:2603.26993.
  </footer>
</div>
</body>
</html>"""


__all__ = ["render_html"]
