#!/usr/bin/env python3
"""Benchmark vLLM at different gpu-memory-utilization values.

Usage:
    python tests/bench_vram.py --util 0.70
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PDF_70 = "test_data/pdfs/2022統合報告書.pdf"
PDF_12 = "test_data/pdfs/2022統合報告書-1-12.pdf"
API_URL = "http://localhost:8000/api/v1/pdf/process"


def wait_for_health(url, timeout=600):
    for _ in range(timeout // 5):
        try:
            r = subprocess.run(
                ["curl", "-s", url], capture_output=True, text=True, timeout=5
            )
            if "healthy" in r.stdout:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip())


def monitor_vram(samples, stop_event):
    while not stop_event.is_set():
        samples.append((time.time(), get_vram()))
        time.sleep(0.5)


def run_pdf(pdf_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = f"{tmpdir}/out.zip"
        t0 = time.time()
        subprocess.run(
            ["curl", "-s", "-X", "POST", API_URL,
             "-F", f"file=@{pdf_path}", "-F", "skip_layouts=true",
             "-o", zip_path],
            capture_output=True, timeout=600,
        )
        wall = time.time() - t0
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", f"{tmpdir}/out"], capture_output=True)
        metrics_path = Path(f"{tmpdir}/out/parsing_metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            overall = metrics["timing"]["overall"]
            return wall, overall["duration_seconds"]
        return wall, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--util", type=float, required=True)
    args = parser.parse_args()

    util = args.util
    print(f"\n{'='*60}")
    print(f"Testing gpu-memory-utilization = {util}")
    print(f"{'='*60}")

    # Start vLLM
    print("Starting vLLM...")
    subprocess.Popen(
        f"source .venv/bin/activate && nohup vllm serve zai-org/GLM-OCR "
        f"--port 8080 --served-model-name glm-ocr "
        f"--speculative-config '{{\"method\": \"mtp\", \"num_speculative_tokens\": 1}}' "
        f"--allowed-local-media-path / "
        f"--gpu-memory-utilization {util} "
        f"> /tmp/vllm_{util}.log 2>&1",
        shell=True, executable="/bin/bash",
    )

    if not wait_for_health("http://localhost:8080/health"):
        print("ERROR: vLLM failed to start")
        sys.exit(1)

    vram_vllm = get_vram()
    print(f"vLLM VRAM: {vram_vllm} MiB ({vram_vllm/1024:.1f} GiB)")

    # Start API
    print("Starting API server...")
    subprocess.Popen(
        "source .venv/bin/activate && nohup python -m uvicorn "
        "src.glm_hybrid_ocr.api.main:app --host 0.0.0.0 --port 8000 "
        f"> /tmp/uvicorn_{util}.log 2>&1",
        shell=True, executable="/bin/bash",
    )

    if not wait_for_health("http://localhost:8000/health"):
        print("ERROR: API failed to start")
        sys.exit(1)

    vram_api = get_vram()
    print(f"API loaded VRAM: {vram_api} MiB (layout detector: +{vram_api - vram_vllm} MiB)")

    # Warm up
    print("\nWarm-up run (70-page)...")
    wall, proc = run_pdf(PDF_70)
    print(f"  Warm-up: wall={wall:.1f}s, processing={proc}s")

    # Single PDF benchmark
    print("\nSingle 70-page PDF (warm)...")
    samples = []
    stop = threading.Event()
    t = threading.Thread(target=monitor_vram, args=(samples, stop), daemon=True)
    t.start()
    wall, proc = run_pdf(PDF_70)
    stop.set()
    t.join(timeout=2)

    peak = max(s[1] for s in samples) if samples else 0
    print(f"  Wall: {wall:.1f}s, Processing: {proc}s")
    print(f"  VRAM idle: {vram_api} MiB, peak: {peak} MiB, delta: +{peak - vram_api} MiB")
    print(f"  Free at peak: {49140 - peak} MiB ({(49140 - peak)/1024:.1f} GiB)")

    # KV cache info
    try:
        log = Path(f"/tmp/vllm_{util}.log").read_bytes()
        import re
        for line in log.decode("utf-8", errors="ignore").split("\n"):
            if "KV cache size" in line or "Maximum concurrency" in line:
                print(f"  {line.strip().split('] ')[-1]}")
    except Exception:
        pass

    print(f"\n--- Summary for util={util} ---")
    print(f"  vLLM VRAM:      {vram_vllm} MiB ({vram_vllm/1024:.1f} GiB)")
    print(f"  Peak VRAM:      {peak} MiB ({peak/1024:.1f} GiB)")
    print(f"  Processing:     {proc}s")
    print(f"  Free for concurrent: {49140 - peak} MiB ({(49140 - peak)/1024:.1f} GiB)")


if __name__ == "__main__":
    main()
