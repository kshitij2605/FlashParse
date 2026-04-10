#!/usr/bin/env python3
"""Benchmark misc OCR strategies on the 70-page PDF.

Usage:
    python tests/benchmark_misc_ocr.py [--runs N] [--label LABEL]

Runs the 70-page PDF N times (default 2) and prints timing for each run.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PDF_PATH = "test_data/pdfs/2022統合報告書.pdf"
API_URL = "http://localhost:8000/api/v1/pdf/process"


def run_once(run_num: int) -> dict:
    """Run one benchmark and return parsed metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "output.zip"
        out_dir = Path(tmpdir) / "output"

        t0 = time.time()
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", API_URL, "-F", f"file=@{PDF_PATH}", "-o", str(zip_path)],
            capture_output=True, text=True, timeout=600,
        )
        wall_time = time.time() - t0

        if result.returncode != 0:
            print(f"  Run {run_num}: curl failed: {result.stderr}")
            return {}

        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)], capture_output=True)

        metrics_path = out_dir / "parsing_metrics.json"
        if not metrics_path.exists():
            print(f"  Run {run_num}: no parsing_metrics.json found")
            return {}

        with open(metrics_path) as f:
            metrics = json.load(f)

        timing = metrics["timing"]
        stats = metrics["statistics"]

        return {
            "run": run_num,
            "wall_time": round(wall_time, 1),
            "timing": timing,
            "num_images": stats["num_images"],
            "num_captions": stats["num_captions_generated"],
        }


def print_result(r: dict, label: str):
    if not r:
        return
    t = r["timing"]
    print(f"\n  [{label}] Run {r['run']} (wall: {r['wall_time']}s)")

    for step, v in t.items():
        if isinstance(v, dict):
            print(f"    {step:25s} {v['start']} -> {v['end']}  {v['duration_formatted']:>8s}")
        else:
            print(f"    {step:25s} {v}")

    imgs = r["num_images"]
    print(f"    images: {imgs['total']} (misc={imgs.get('num_miscellaneous_images', '?')}, "
          f"chart={imgs.get('num_chart_images', '?')}, figure={imgs.get('num_figure_images', '?')}, "
          f"scanned={imgs.get('num_scanned_text_images', '?')})")
    print(f"    captions generated: {r['num_captions']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--label", default="benchmark")
    args = parser.parse_args()

    if not Path(PDF_PATH).exists():
        print(f"ERROR: {PDF_PATH} not found")
        sys.exit(1)

    print(f"=== Misc OCR Benchmark: {args.label} ===")
    print(f"PDF: {PDF_PATH}")
    print(f"Runs: {args.runs}")

    results = []
    for i in range(1, args.runs + 1):
        print(f"\n--- Run {i}/{args.runs} ---")
        r = run_once(i)
        results.append(r)
        print_result(r, args.label)

    # Summary
    if len(results) >= 2 and results[-1]:
        print(f"\n=== WARM RUN RESULT ({args.label}) ===")
        t = results[-1]["timing"]
        overall = t.get("overall", {})
        if isinstance(overall, dict):
            print(f"  Total: {overall.get('duration_formatted', '?')}")


if __name__ == "__main__":
    main()
