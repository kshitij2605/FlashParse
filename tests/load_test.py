#!/usr/bin/env python3
"""Simple load test: send N concurrent PDF requests and measure timing.

Usage:
    python tests/load_test.py --concurrency 1
    python tests/load_test.py --concurrency 2
    python tests/load_test.py --concurrency 3
"""
import argparse
import asyncio
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000/api/v1/pdf/process"
PDF_12 = "test_data/pdfs/2022統合報告書-1-12.pdf"


async def send_request(client: httpx.AsyncClient, pdf_path: str, req_id: int) -> dict:
    """Send one PDF processing request, return timing info."""
    t0 = time.time()
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        try:
            response = await client.post(API_URL, files=files)
            elapsed = time.time() - t0
            ok = response.status_code == 200
            size = len(response.content) if ok else 0
            return {
                "id": req_id, "status": response.status_code,
                "elapsed": round(elapsed, 1), "ok": ok, "size_kb": round(size / 1024),
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "id": req_id, "status": "error",
                "elapsed": round(elapsed, 1), "ok": False, "error": str(e),
            }


async def run_load_test(concurrency: int, pdf_path: str):
    print(f"\n=== Load Test: {concurrency} concurrent request(s) ===")
    print(f"PDF: {pdf_path}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        # Warm up with 1 request first
        print("Warming up (1 request)...")
        warm = await send_request(client, pdf_path, 0)
        print(f"  Warm-up: {warm['elapsed']}s, status={warm['status']}")

        # Send N concurrent requests
        print(f"\nSending {concurrency} concurrent requests...")
        t_start = time.time()
        tasks = [
            asyncio.create_task(send_request(client, pdf_path, i + 1))
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        t_total = time.time() - t_start

        # Report
        print(f"\n{'ID':>4} {'Status':>8} {'Time':>8} {'Size':>8}")
        print("-" * 32)
        for r in sorted(results, key=lambda x: x["id"]):
            status = "OK" if r["ok"] else str(r.get("status", "err"))
            size = f"{r.get('size_kb', 0)}KB" if r["ok"] else r.get("error", "")[:30]
            print(f"{r['id']:>4} {status:>8} {r['elapsed']:>7.1f}s {size:>8}")

        ok_results = [r for r in results if r["ok"]]
        fail_count = concurrency - len(ok_results)
        if ok_results:
            times = [r["elapsed"] for r in ok_results]
            print(f"\n  Concurrent: {concurrency}")
            print(f"  All completed in: {t_total:.1f}s")
            print(f"  Fastest: {min(times):.1f}s | Slowest: {max(times):.1f}s | Avg: {sum(times)/len(times):.1f}s")
            print(f"  Throughput: {len(ok_results)/t_total:.2f} req/s")
            if fail_count:
                print(f"  Failed: {fail_count}/{concurrency}")
        else:
            print(f"\n  All {concurrency} requests failed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", "-c", type=int, default=1)
    parser.add_argument("--pdf", default=PDF_12)
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"ERROR: {args.pdf} not found")
        return

    asyncio.run(run_load_test(args.concurrency, args.pdf))


if __name__ == "__main__":
    main()
