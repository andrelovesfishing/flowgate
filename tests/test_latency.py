import torch

from flowgate.latency import ms_per_query


def test_times_one_query_at_a_time_on_one_thread_and_restores_threads():
    seen, threads = [], []
    before = torch.get_num_threads()

    def fn(q):
        seen.append(q.shape); threads.append(torch.get_num_threads())
        return q * 2
    ms = ms_per_query(fn, torch.randn(30, 4), n=10, warmup=2)
    assert ms > 0 and set(seen) == {(1, 4)} and len(seen) == 12 and set(threads) == {1}
    assert torch.get_num_threads() == before


def test_interleaved_timing_separates_slow_from_fast_one_query_one_thread():
    import time

    from flowgate.latency import time_interleaved
    before, seen = torch.get_num_threads(), {"slow": [], "fast": []}

    def slow(q):
        seen["slow"].append((q.shape, torch.get_num_threads())); time.sleep(0.003)

    def fast(q):
        seen["fast"].append((q.shape, torch.get_num_threads()))
    out = time_interleaved({"slow": slow, "fast": fast}, torch.randn(10, 4), n=4, rounds=3, warmup=1)
    assert out["slow"]["median"] >= 2.5 and out["fast"]["median"] < 1.0
    assert out["slow"]["n"] == 12 and len(seen["slow"]) == 12 + 1  # 1 warmup + 3 rounds x 4
    assert {s for s, _ in seen["fast"]} == {(1, 4)} and {t for _, t in seen["fast"]} == {1}
    assert out["slow"]["p10"] <= out["slow"]["median"] <= out["slow"]["p90"]
    assert torch.get_num_threads() == before


def test_every_round_visits_every_method_so_machine_drift_hits_all_alike():
    from flowgate.latency import time_interleaved
    order = []
    fns = {k: (lambda q, k=k: order.append(k)) for k in "abc"}
    time_interleaved(fns, torch.randn(5, 2), n=2, rounds=4, warmup=0)
    blocks = order[::2]  # each method runs n=2 queries in a row
    assert len(blocks) == 12 and all(set(blocks[3 * r : 3 * r + 3]) == set("abc") for r in range(4))
