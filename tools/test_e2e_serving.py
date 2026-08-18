#!/usr/bin/env python3
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def wait_for_port(host: str, port: int, timeout: float = 180.0) -> bool:
    start = time.time()
    url = f"http://{host}:{port}/health"
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def send_chat_completion(port: int, prompt: str, max_tokens: int = 32) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "qwen3_5_9b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        t1 = time.perf_counter()
        body = json.loads(resp.read().decode("utf-8"))
        elapsed = t1 - t0
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(content.split()))
        return {
            "elapsed": elapsed,
            "tokens": completion_tokens,
            "content": content,
            "tok_per_sec": completion_tokens / max(elapsed, 1e-4),
        }


def run_concurrency_test(model_path: str, concurrency: int, port: int = 8090, max_tokens: int = 256):
    cmd = [
        "./build/apps/ninfer-serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-concurrency",
        str(concurrency),
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--model-id",
        "qwen3_5_9b",
        "--default-max-tokens",
        str(max_tokens),
    ]
    print(f"\n========================================================")
    print(f"Starting server: concurrency={concurrency}, MTP=3, max_tokens={max_tokens}, port={port}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        print("Waiting for server to become ready...")
        if not wait_for_port("127.0.0.1", port, timeout=150.0):
            print("ERROR: Server timed out starting up!")
            proc.terminate()
            out, _ = proc.communicate(timeout=5)
            print(out)
            return None

        print(f"Server ready on port {port}. Sending {concurrency} concurrent request(s)...")

        prompts = [
            f"Explain the principles of thermodynamics, entropy, and statistical mechanics in detail. Query index {i+1}."
            for i in range(concurrency)
        ]

        t_batch_start = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(send_chat_completion, port, prompts[i], max_tokens)
                for i in range(concurrency)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        t_batch_end = time.perf_counter()

        batch_elapsed = t_batch_end - t_batch_start
        total_tokens = sum(r["tokens"] for r in results)
        aggregate_throughput = total_tokens / max(batch_elapsed, 1e-4)

        print(f"\n--- Concurrency {concurrency} Results ---")
        print(f"Batch elapsed: {batch_elapsed:.3f} s")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Aggregate throughput: {aggregate_throughput:.2f} tok/s")
        for idx, res in enumerate(results):
            print(f"  Req {idx+1}: {res['tokens']} tokens in {res['elapsed']:.3f}s ({res['tok_per_sec']:.2f} tok/s) -> {res['content'].strip()[:60]}...")

        return {
            "concurrency": concurrency,
            "batch_elapsed": batch_elapsed,
            "total_tokens": total_tokens,
            "aggregate_tok_per_sec": aggregate_throughput,
            "results": results,
        }

    finally:
        print(f"Stopping server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Serving Concurrency Test")
    parser.add_argument("--model", default="models/qwen3_5_9b.ninfer", help="Path to .ninfer model")
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4, 8], help="Concurrency levels to test")
    parser.add_argument("--port", type=int, default=8090, help="Base port for serving")
    parser.add_argument("--max-tokens", type=int, default=256, help="Tokens to generate per stream")
    args = parser.parse_args()

    all_summaries = []
    for c in args.concurrencies:
        summary = run_concurrency_test(args.model, concurrency=c, port=args.port, max_tokens=args.max_tokens)
        if summary is not None:
            all_summaries.append(summary)
        time.sleep(2)

    print("\n========================================================")
    print("                 FINAL SUMMARY TABLE                    ")
    print("========================================================")
    print(f"{'Concurrency':<12} | {'Batch Time (s)':<15} | {'Tokens':<8} | {'Throughput (tok/s)':<20}")
    print("-" * 65)
    for s in all_summaries:
        print(f"{s['concurrency']:<12} | {s['batch_elapsed']:<15.3f} | {s['total_tokens']:<8} | {s['aggregate_tok_per_sec']:<20.2f}")
    print("========================================================")


if __name__ == "__main__":
    main()
