#!/usr/bin/env python3
"""Read tpq_report.json files from a sweep and print a table of
adaptive Krylov diagnostics (min_m / max_m / mean_m) vs the swept
parameter.

Usage:
    python3 scripts/analyze_adaptive_m.py <axis> <runs_glob>

Where:
    <axis>      One of "beta" | "tol" | "samples". Used as the column
                label and to parse the swept value out of the run dir
                basename (after the sweep_adaptive_m_<axis>_ prefix).
    <runs_glob> Glob matching run directories holding tpq_report.json.
                Example: "runs/sweep_adaptive_m_beta_*"

For each matched run, prints a row with:
    <axis>     min_m   max_m   mean_m    wall_ms

Run dir naming convention (set by configs/sweeps/*.toml):
    sweep_adaptive_m_beta_<value-with-dot-as-p>
        e.g.  sweep_adaptive_m_beta_0p5      -> 0.5
    sweep_adaptive_m_tol_<scientific>
        e.g.  sweep_adaptive_m_tol_1e-10     -> 1e-10
    sweep_adaptive_m_samples_<int>
        e.g.  sweep_adaptive_m_samples_500   -> 500
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys


def parse_value(name: str, axis: str):
    match = re.search(rf"sweep_adaptive_m_{axis}_(.+)$", name)
    if match is None:
        return None
    raw = match.group(1)
    if axis == "beta":
        # Numbers encoded with 'p' as decimal separator: 0p5 -> 0.5
        return float(raw.replace("p", "."))
    if axis == "tol":
        return float(raw)
    if axis == "samples":
        return int(raw)
    raise SystemExit("unknown axis %r; expected beta | tol | samples" % axis)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    axis = sys.argv[1]
    pattern = sys.argv[2]
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print("no runs matched %r" % pattern, file=sys.stderr)
        return 1

    rows = []
    for d in run_dirs:
        run = pathlib.Path(d)
        report = run / "tpq_report.json"
        if not report.exists():
            print("skipping %s: no tpq_report.json" % d, file=sys.stderr)
            continue
        value = parse_value(run.name, axis)
        if value is None:
            print("skipping %s: name does not match sweep prefix" % d, file=sys.stderr)
            continue
        with report.open() as f:
            payload = json.load(f)
        stats = payload.get("krylov_stats")
        if stats is None:
            print("skipping %s: tpq_report.json has no krylov_stats" % d, file=sys.stderr)
            continue
        rows.append((
            value,
            int(stats["min_m"]),
            int(stats["max_m"]),
            float(stats["mean_m"]),
            int(payload["wall_time_ms"]),
        ))

    rows.sort()
    print("%10s %6s %6s %8s %8s" % (axis, "min_m", "max_m", "mean_m", "wall_ms"))
    print("-" * 44)
    for value, min_m, max_m, mean_m, wall_ms in rows:
        if axis == "tol":
            value_str = "%.0e" % value
        else:
            value_str = str(value)
        print("%10s %6d %6d %8.1f %8d" % (value_str, min_m, max_m, mean_m, wall_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
