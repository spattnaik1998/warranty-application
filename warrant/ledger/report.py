"""Render the ledger scorecard and reproduce the paper's plots."""

from __future__ import annotations

from pathlib import Path

from warrant.config import get_settings
from warrant.ledger.probe import ProbeResult
from warrant.schemas.ledger import Condition


def acceptance(probe: ProbeResult) -> dict[str, bool]:
    """The acceptance criteria from the plan, evaluated on this probe."""
    a = probe.runs[Condition.CENTRALIZED].accuracy
    b = probe.runs[Condition.GOVERNED].accuracy
    naive = probe.runs[Condition.NAIVE].accuracy
    starved = probe.runs[Condition.SIGNAL_STARVED].accuracy
    return {
        "governed matches centralized (B ≈ A)": b >= a - 1e-6,
        "naive collapses below governed (B- < B)": naive < b,
        "signal-starved drops below governed (C < B)": starved < b,
        "KL predicts accuracy drop (r > 0.5)": probe.kl_accuracy_r > 0.5,
        "prose degrades faster than posterior interface":
            probe.prose_slope() > probe.posterior_slope(),
    }


def render_report(probe: ProbeResult) -> str:
    lines: list[str] = ["# Delegation Ledger — reliability scorecard", ""]
    lines.append("## Matched conditions")
    lines.append("")
    lines.append("| Condition | Accuracy | Comm. loss | Depth | Admitted | Rejected | Redundant |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    order = [Condition.CENTRALIZED, Condition.GOVERNED, Condition.NAIVE, Condition.SIGNAL_STARVED]
    for cond in order:
        r = probe.runs.get(cond)
        if not r:
            continue
        lines.append(
            f"| {cond.value} | {r.accuracy:.3f} | {r.communication_loss:.4f} | "
            f"{r.delegation_depth} | {r.admitted_delegations} | "
            f"{r.rejected_delegations} | {r.redundant_delegations} |"
        )
    lines += ["", "## Relay depth sweep (prose interface)", "",
              "| Depth | Accuracy | Mean KL vs centralized | Accuracy drop |",
              "|---:|---:|---:|---:|"]
    for p in probe.sweep:
        if p.interface == "prose":
            lines.append(f"| {p.depth} | {p.accuracy:.3f} | {p.mean_kl:.3f} | {p.accuracy_drop:.3f} |")

    lines += ["", "## Headline results", "",
              f"- **KL ↔ accuracy-drop correlation (r):** {probe.kl_accuracy_r:.3f} "
              f"(paper reports r≈0.72)",
              f"- **Prose relay degradation:** {probe.prose_slope():.1f} accuracy pts/stage",
              f"- **Posterior-interface degradation:** {probe.posterior_slope():.1f} pts/stage",
              "", "## Acceptance criteria", ""]
    for name, ok in acceptance(probe).items():
        lines.append(f"- [{'x' if ok else ' '}] {name}")
    lines.append("")
    return "\n".join(lines)


def _plots(probe: ProbeResult, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    # 1) Accuracy vs relay depth, prose vs posterior interface.
    fig, ax = plt.subplots(figsize=(6, 4))
    for interface, color in (("prose", "#c0392b"), ("posterior", "#2471a3")):
        pts = [p for p in probe.sweep if p.interface == interface]
        ax.plot([p.depth for p in pts], [p.accuracy * 100 for p in pts],
                marker="o", label=f"{interface} relay", color=color)
    ax.axhline(25, ls="--", color="grey", lw=1, label="random (4-way)")
    ax.set_xlabel("relay depth (reorganizer hops)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Telephone game: accuracy vs relay depth")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    p1 = outdir / "accuracy_vs_depth.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(p1)

    # 2) KL vs accuracy drop (prose).
    fig, ax = plt.subplots(figsize=(6, 4))
    prose = [p for p in probe.sweep if p.interface == "prose"]
    ax.scatter([p.mean_kl for p in prose], [p.accuracy_drop * 100 for p in prose],
               color="#c0392b")
    for p in prose:
        ax.annotate(f"d={p.depth}", (p.mean_kl, p.accuracy_drop * 100),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("mean KL(centralized ‖ relayed)  [bits]")
    ax.set_ylabel("accuracy drop (pts)")
    ax.set_title(f"KL predicts accuracy loss (r = {probe.kl_accuracy_r:.2f})")
    fig.tight_layout()
    p2 = outdir / "kl_vs_accuracy_drop.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths.append(p2)

    # 3) Accuracy by condition.
    fig, ax = plt.subplots(figsize=(6, 4))
    order = [Condition.CENTRALIZED, Condition.GOVERNED, Condition.NAIVE, Condition.SIGNAL_STARVED]
    labels = ["A\ncentralized", "B\ngoverned", "B-\nnaive relay", "C\nsignal-starved"]
    vals = [probe.runs[c].accuracy * 100 for c in order if c in probe.runs]
    colors = ["#1e8449", "#2471a3", "#c0392b", "#b9770e"]
    ax.bar(labels[:len(vals)], vals, color=colors[:len(vals)])
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy by condition")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p3 = outdir / "accuracy_by_condition.png"
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    paths.append(p3)
    return paths


def write_report(probe: ProbeResult, outdir: str | Path | None = None) -> Path:
    """Write the scorecard markdown and the three PNG plots; return the md path."""
    out = Path(outdir) if outdir else get_settings().output_dir
    out.mkdir(parents=True, exist_ok=True)
    md = render_report(probe)
    plot_paths = _plots(probe, out)
    md += "\n## Figures\n\n" + "\n".join(f"![{p.stem}]({p.name})" for p in plot_paths) + "\n"
    report_path = out / "ledger_report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
