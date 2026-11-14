"""Per-query latency under one fixed protocol: one query at a time, one CPU thread, median.

A gate answers one query at a time on a phone, so batch-1 is the honest unit. One thread keeps
numbers comparable across runs (the laptop's own load and battery state otherwise swing timings
2x). The hardware-free cost, network calls per query, is reported next to it.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch


def ms_per_query(fn, Z: torch.Tensor, n: int = 20, warmup: int = 2) -> float:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for i in range(warmup):
            fn(Z[i : i + 1])
        times = []
        for i in range(n):
            q = Z[(warmup + i) % len(Z)][None]
            t0 = time.perf_counter(); fn(q); times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times)
    finally:
        torch.set_num_threads(old)


def time_interleaved(fns: dict, X: torch.Tensor, n: int = 20, rounds: int = 5, warmup: int = 3,
                     seed: int = 0) -> dict[str, dict]:
    """Time several methods under the same protocol, interleaved: in each round every method runs n
    batch-1 queries, in a fresh random order. Slow drift in the machine's state (heat, power, other
    load) then lands on every method alike instead of on whichever ran last. Returns, per method,
    the median and 10th/90th percentiles in ms over all n x rounds queries."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    names, times = list(fns), {k: [] for k in fns}
    try:
        with torch.no_grad():
            for k in names:
                for i in range(warmup):
                    fns[k](X[i % len(X)][None])
            for _ in range(rounds):
                for k in rng.permutation(names):
                    for i in range(n):
                        q = X[(len(times[k]) + i) % len(X)][None]
                        t0 = time.perf_counter(); fns[k](q); times[k].append(time.perf_counter() - t0)
    finally:
        torch.set_num_threads(old)
    return {k: {"median": 1e3 * float(np.median(v)), "p10": 1e3 * float(np.percentile(v, 10)),
                "p90": 1e3 * float(np.percentile(v, 90)), "n": len(v)} for k, v in times.items()}
