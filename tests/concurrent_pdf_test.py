"""Test concurrent PDF processing with the API server.

Sends N concurrent requests to the API and reports timing/failures.
Requires API server running at localhost:8000.
"""

import asyncio
import sys
import time

import httpx

API_URL = "http://localhost:8000/api/v1/pdf/process"
PDF_PATH = "/home/mac/25gitlab/notebooklm/notebook_data/uploads/2022統合報告書.pdf"


async def send_pdf(client: httpx.AsyncClient, idx: int) -> dict:
    """Send a single PDF processing request."""
    t0 = time.time()
    try:
        with open(PDF_PATH, "rb") as f:
            response = await client.post(
                API_URL,
                files={"file": ("test.pdf", f, "application/pdf")},
                data={"skip_captions": "true", "skip_layouts": "true"},
            )
        elapsed = time.time() - t0
        if response.status_code == 200:
            pages = response.headers.get("X-Pages-Processed", "?")
            return {"idx": idx, "status": "ok", "time": elapsed, "pages": pages}
        else:
            return {"idx": idx, "status": "error", "time": elapsed, "code": response.status_code, "body": response.text[:500]}
    except Exception as e:
        elapsed = time.time() - t0
        return {"idx": idx, "status": "exception", "time": elapsed, "error": str(e)[:500]}


async def test_concurrent(n: int):
    print(f"\n{'='*60}")
    print(f"Testing {n} concurrent PDF requests (70-page PDF, skip_captions)")
    print(f"{'='*60}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        t0 = time.time()
        tasks = [send_pdf(client, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - t0

    print(f"\nWall time: {wall_time:.1f}s")
    for r in sorted(results, key=lambda x: x["idx"]):
        if r["status"] == "ok":
            print(f"  Request {r['idx']}: OK in {r['time']:.1f}s ({r['pages']} pages)")
        elif r["status"] == "error":
            print(f"  Request {r['idx']}: HTTP {r['code']} in {r['time']:.1f}s — {r['body']}")
        else:
            print(f"  Request {r['idx']}: EXCEPTION in {r['time']:.1f}s — {r['error']}")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"\nResult: {ok_count}/{n} succeeded")
    return ok_count == n


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    success = await test_concurrent(n)
    if success:
        print(f"\n*** {n} concurrent requests: PASSED ***")
    else:
        print(f"\n*** FAILED at {n} concurrent requests ***")


if __name__ == "__main__":
    asyncio.run(main())
