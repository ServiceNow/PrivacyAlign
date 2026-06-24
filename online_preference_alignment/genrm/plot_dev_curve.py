"""Render an ASCII curve of Phase A dev-eval metrics over training steps.

Usage:
    python -m genrm.plot_dev_curve [--log outputs/genrm-privalign-qwen3-4b/training_log.txt]

Reads:
  - `Pairwise-margin dev eval: pair_acc=... signed_mean=... invalid_rate=...`
    lines emitted by ``_run_pairwise_margin_dev_eval``.
  - The most recent ``step=<N> train`` line preceding each dev-eval line gives
    the step number, so the chart is plotted against the actual prompt-step.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_DEV_RE = re.compile(
    r"Pairwise-margin dev eval: "
    r"pair_acc=(?P<pa>[-+0-9.eE]+) "
    r"signed_mean=(?P<sm>[-+0-9.eE]+) "
    r"invalid_rate=(?P<iv>[-+0-9.eE]+) "
    r"n_valid=(?P<nv>\d+) "
    r"n_invalid=(?P<ni>\d+)"
)
_STEP_RE = re.compile(r"\[INFO\] step=(?P<step>\d+) train")


def parse_log(path: Path) -> list[dict]:
    """Return list of dev-eval records: {step, pair_acc, signed_mean, invalid_rate}."""
    rows: list[dict] = []
    last_step = 0
    with path.open() as f:
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                last_step = int(m.group("step"))
                continue
            m = _DEV_RE.search(line)
            if m:
                rows.append(
                    {
                        "step": last_step,
                        "pair_acc": float(m.group("pa")),
                        "signed_mean": float(m.group("sm")),
                        "invalid_rate": float(m.group("iv")),
                        "n_valid": int(m.group("nv")),
                        "n_invalid": int(m.group("ni")),
                    }
                )
    return rows


def render_curve(
    rows: list[dict],
    *,
    metric: str,
    label: str,
    vmin: float,
    vmax: float,
    width: int = 78,
    height: int = 14,
) -> str:
    if not rows:
        return f"[no dev-eval rows yet for {metric}]"
    xs = [r["step"] for r in rows]
    ys = [r[metric] for r in rows]

    def y_row(v: float) -> int:
        f = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        return height - 1 - int(round(f * (height - 1)))

    n = len(rows)
    def x_col(i: int) -> int:
        return 0 if n == 1 else int(round(i * (width - 1) / (n - 1)))

    grid = [[" "] * width for _ in range(height)]

    # Zero line for signed_mean charts.
    if vmin < 0 < vmax:
        zr = y_row(0.0)
        for c in range(width):
            grid[zr][c] = "-"

    # Points + connecting lines (vertical fill).
    prev_r = None
    prev_c = None
    for i, v in enumerate(ys):
        r = y_row(v)
        c = x_col(i)
        if prev_r is not None and prev_c is not None and prev_c != c:
            # connect with vertical bars + horizontal dashes between points
            r1, r2 = sorted([prev_r, r])
            mid_c = (prev_c + c) // 2
            for rr in range(r1, r2 + 1):
                if grid[rr][mid_c] == " ":
                    grid[rr][mid_c] = "."
        grid[r][c] = "*"
        prev_r = r
        prev_c = c

    out = [f"{label}  (steps {xs[0]} -> {xs[-1]}; y in [{vmin}, {vmax}])"]
    out.append("+" + "-" * width + "+")
    for r, row in enumerate(grid):
        v_at = vmax - (vmax - vmin) * r / (height - 1)
        out.append(f"|{''.join(row)}| {v_at:+.3f}")
    out.append("+" + "-" * width + "+")
    # x-axis tick labels (steps for first, middle, last point)
    tick_idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    labels = [" "] * width
    for ti in tick_idxs:
        s = str(xs[ti])
        c = x_col(ti)
        start = max(0, c - len(s) // 2)
        for k, ch in enumerate(s):
            if start + k < width:
                labels[start + k] = ch
    out.append(" " + "".join(labels))
    out.append("")
    out.append(
        f"  most_recent_step={xs[-1]}  {metric}={ys[-1]:+.3f}  "
        f"first={ys[0]:+.3f}  delta={ys[-1] - ys[0]:+.3f}  "
        f"min={min(ys):+.3f}  max={max(ys):+.3f}  n={n}"
    )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="outputs/genrm-privalign-qwen3-4b/training_log.txt",
        help="Path to the Phase A training log.",
    )
    args = parser.parse_args()
    rows = parse_log(Path(args.log))
    if not rows:
        print("No dev-eval lines parsed yet. Wait until step >= dev_eval_steps.")
        return
    print(render_curve(rows, metric="pair_acc", label="dev/pair_accuracy", vmin=0.0, vmax=1.0))
    print()
    print(render_curve(rows, metric="signed_mean", label="dev/signed_target_mean", vmin=-2.0, vmax=2.0))
    print()
    print(
        render_curve(
            rows, metric="invalid_rate", label="dev/format_violation_rate", vmin=0.0, vmax=0.5
        )
    )
    print()
    print("Tabular:")
    print("step  pair_acc  signed_mean  invalid_rate  n_valid  n_invalid")
    for r in rows:
        print(
            f"{r['step']:>4}  {r['pair_acc']:>8.3f}  {r['signed_mean']:>+11.3f}  "
            f"{r['invalid_rate']:>12.3f}  {r['n_valid']:>7}  {r['n_invalid']:>9}"
        )


if __name__ == "__main__":
    main()
