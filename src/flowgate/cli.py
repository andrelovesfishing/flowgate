"""flowgate CLI.

  r1     train (or reuse) one flow-ratio gate and score it on cal (val + train_oos) or, once, on test
  sweep  run r1 over a grid of d x width on cal and print one table

Trained models are cached under $KWA_DATA_DIR/flowgate/models/, keyed by training settings only,
so rescoring (other ODE steps, other eval sets) never retrains. Baselines are cached per space.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from kwa.backbones import load_cache
from kwa.config import data_dir
from kwa.gates import SplitData
from kwa.metrics import auroc_oos

from .gate import FlowRatioGate
from .space import Space
from .stats import paired_auroc_diff
from .train import TrainConfig

BACKBONE, DATASET = "answerdotai/ModernBERT-base", "clinc150"
N_PROBE = 200


def root() -> Path:
    return data_dir() / "flowgate"


def _split(name: str) -> SplitData:
    z = load_cache(BACKBONE, DATASET, name, data_dir())
    return SplitData(labels=z["label"].astype(int), emb={"frozen": z["emb"].astype(np.float32)})


def eval_set(split: str, subsample: int | None, seed: int) -> SplitData:
    """cal = val (in-scope + OOS) + train_oos; test = test. Subsampling keeps every OOS example."""
    parts = [_split("val"), _split("train_oos")] if split == "cal" else [_split("test")]
    labels = np.concatenate([p.labels for p in parts])
    X = np.concatenate([p.emb["frozen"] for p in parts])
    if subsample:
        rng = np.random.default_rng(seed)
        ins = np.flatnonzero(labels >= 0)
        keep = np.sort(np.r_[rng.choice(ins, min(subsample, len(ins)), replace=False), np.flatnonzero(labels < 0)])
        labels, X = labels[keep], X[keep]
    return SplitData(labels=labels, emb={"frozen": X})


def _eval_tag(a) -> str:
    return f"{a.split}{'-n' + str(a.subsample) if a.subsample else ''}"


def baselines(train: SplitData, ev: SplitData, d: int, kind: str, gmm: bool,
              cache: Path | None = None) -> dict[str, np.ndarray]:
    """Per-point baseline scores on ev (higher = act), cached as .npz."""
    if cache and cache.exists():
        with np.load(cache) as z:
            out = {k: z[k] for k in z.files}
        if not gmm or any(k.startswith("gmm") for k in out):
            return out
    Xtr, Xe = train.emb["frozen"].astype(np.float64), ev.emb["frozen"].astype(np.float64)
    full = Space.fit(Xtr, Xtr.shape[1], "pca").transform(Xe)
    out = {"gaussian": -(full.astype(np.float64) ** 2).sum(1)}
    A = Xe / np.linalg.norm(Xe, axis=1, keepdims=True)
    B = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    out["top1_cosine"] = (A @ B.T).max(1)
    if gmm:
        from sklearn.mixture import GaussianMixture
        print(f"  fitting GMM-150 baseline in {kind}{d} (one-off, cached)...", flush=True)
        sp = Space.fit(Xtr, d, kind)
        ztr, ze = sp.transform(Xtr).astype(np.float64), sp.transform(Xe).astype(np.float64)
        gm = GaussianMixture(150, covariance_type="tied", random_state=0, max_iter=200, reg_covar=1e-3).fit(ztr)
        out[f"gmm150_ratio_{kind}{d}"] = gm.score_samples(ze) + 0.5 * (ze**2).sum(1)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or "uncommitted"
    except OSError:
        return "unknown"


def model_key(a) -> str:
    return f"{a.kind}{a.d}-w{a.width}-L{a.depth}-st{a.steps}-lr{a.lr:g}-s{a.seed}"


def get_gate(a, train: SplitData) -> FlowRatioGate:
    path = root() / "models" / f"{model_key(a)}.pt"
    if path.exists() and not a.retrain:
        print(f"reusing trained model {path.name}", flush=True)
        g = FlowRatioGate.load(path)
        g.ode_steps = a.ode_steps
        return g
    cfg = TrainConfig(steps=a.steps, lr=a.lr, seed=a.seed)
    g = FlowRatioGate(d=a.d, kind=a.kind, width=a.width, depth=a.depth, ode_steps=a.ode_steps, train_cfg=cfg)
    print(f"training {model_key(a)} on {len(train.labels)} unlabeled embeddings...", flush=True)
    g.fit(train, log=lambda s: print(s, flush=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(path)
    return g


def r1(a: argparse.Namespace) -> dict:
    if a.split == "test" and not a.final:
        raise SystemExit("refusing to score test without --final (test is never used to choose anything)")
    train = _split("train")
    ev = eval_set(a.split, a.subsample, a.seed)
    is_oos = ev.labels < 0
    gate = get_gate(a, train)
    print(f"  trained {gate.history[-1]['step']} steps in {gate.fit_seconds:.0f}s; best early-stop CFM loss "
          f"{min(h['val'] for h in gate.history):.4f}", flush=True)
    res = {"config": {k: v for k, v in vars(a).items() if k != "func"}, "model": model_key(a),
           "train_cfg": gate.train_cfg.to_dict(), "code_sha": _git_sha(), "history": gate.history,
           "fit_seconds": gate.fit_seconds, "n_eval": int(len(ev.labels)), "n_oos": int(is_oos.sum())}

    if a.time_probe:
        probe = SplitData(labels=ev.labels[:N_PROBE], emb={"frozen": ev.emb["frozen"][:N_PROBE]})
        t0 = time.perf_counter(); gate.score(probe); per_pt = (time.perf_counter() - t0) / N_PROBE
        res["time_probe"] = {"sec_per_point": per_pt, "est_eval_min": per_pt * len(ev.labels) / 60}
        print(f"oracle likelihood: {per_pt * 1e3:.1f} ms/point ({a.ode_steps} midpoint steps, exact trace); "
              f"this eval set ({len(ev.labels)} pts): ~{per_pt * len(ev.labels) / 60:.1f} min")
    else:
        print(f"scoring {len(ev.labels)} points (exact-trace ODE, {a.ode_steps} steps)...", flush=True)
        t0 = time.perf_counter(); s = gate.score(ev); res["score_seconds"] = time.perf_counter() - t0
        cache = root() / "baselines" / f"{_eval_tag(a)}-{a.kind}{a.d}.npz"
        scores = {gate.name: s, **baselines(train, ev, a.d, a.kind, gmm=not a.no_gmm, cache=cache)}
        res["auroc"] = {k: auroc_oos(v, is_oos) for k, v in scores.items()}
        res["paired_vs_flow"] = {k: paired_auroc_diff(s, v, is_oos) for k, v in scores.items() if k != gate.name}
        print(f"\nOOS AUROC on {a.split} ({len(ev.labels)} pts, {int(is_oos.sum())} OOS); scoring took "
              f"{res['score_seconds'] / 60:.1f} min")
        for k, v in sorted(res["auroc"].items(), key=lambda kv: -kv[1]):
            p = res["paired_vs_flow"].get(k)
            ci = f"   flow - this: {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]" if p else ""
            print(f"  {k:28s} {v:.3f}{ci}")
        print("  reference: labelled relmaha on full cal 0.950 (ceiling); CIs are paired stratified bootstrap 95%")

    run_id = f"r1-{model_key(a)}-ode{a.ode_steps}-{_eval_tag(a)}{'-probe' if a.time_probe else ''}"
    out = root() / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
    if not a.time_probe:  # raw per-point scores are the source of truth for any later analysis
        np.savez(out / "scores.npz", labels=ev.labels, **scores)
    print(f"wrote {out / 'result.json'}\n", flush=True)
    return res


def sweep(a: argparse.Namespace) -> list[dict]:
    rows = []
    for d in a.ds:
        for w in a.widths:
            b = copy.copy(a); b.d, b.width, b.time_probe = d, w, False
            print(f"=== sweep: d={d} width={w} ===", flush=True)
            res = r1(b)
            gk = next((k for k in res["auroc"] if k.startswith("gmm")), None)
            gm = res["auroc"][gk] if gk else float("nan")
            ci = res["paired_vs_flow"][gk] if gk else {"lo": float("nan"), "hi": float("nan")}
            rows.append({"d": d, "width": w, "flow": res["auroc"][f"flowratio-{a.kind}{d}"], "gmm_same_space": gm,
                         "flow_minus_gmm_lo": ci["lo"], "flow_minus_gmm_hi": ci["hi"],
                         "top1_cosine": res["auroc"]["top1_cosine"], "train_min": res["fit_seconds"] / 60,
                         "score_min": res["score_seconds"] / 60})
    print(f"\nSWEEP on {_eval_tag(a)} ({a.kind}, depth {a.depth}, {a.steps} steps). Ceiling: relmaha 0.950.")
    print(f"{'d':>5} {'width':>6} {'flow':>7} {'gmm':>7} {'flow-gmm [95% CI]':>26} {'top1cos':>8} {'train m':>8} {'score m':>8}")
    for r in rows:
        ci = f"{r['flow'] - r['gmm_same_space']:+.3f} [{r['flow_minus_gmm_lo']:+.3f}, {r['flow_minus_gmm_hi']:+.3f}]"
        print(f"{r['d']:>5} {r['width']:>6} {r['flow']:>7.3f} {r['gmm_same_space']:>7.3f} {ci:>26} "
              f"{r['top1_cosine']:>8.3f} {r['train_min']:>8.1f} {r['score_min']:>8.1f}")
    out = root() / "sweeps" / f"sweep-{_eval_tag(a)}-{a.kind}-L{a.depth}-st{a.steps}-s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return rows


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kind", choices=["pca", "wpca"], default="pca")
    common.add_argument("--depth", type=int, default=4)
    common.add_argument("--steps", type=int, default=10_000)
    common.add_argument("--lr", type=float, default=1e-3)
    common.add_argument("--ode-steps", type=int, default=32)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--subsample", type=int, default=None, help="in-scope eval points to keep (all OOS kept)")
    common.add_argument("--split", choices=["cal", "test"], default="cal")
    common.add_argument("--final", action="store_true", help="required for --split test; use once")
    common.add_argument("--no-gmm", action="store_true", help="skip the (slow, one-off) GMM-150 baseline")
    common.add_argument("--retrain", action="store_true", help="ignore a cached trained model")

    p = argparse.ArgumentParser(prog="flowgate")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("r1", parents=[common], help="train or reuse one flow-ratio gate and score it")
    q.add_argument("--d", type=int, default=128)
    q.add_argument("--width", type=int, default=512)
    q.add_argument("--time-probe", action="store_true", help="time the oracle on 200 points instead of scoring")
    q.set_defaults(func=r1)
    s = sub.add_parser("sweep", parents=[common], help="r1 over d x width on cal, one summary table")
    s.add_argument("--ds", type=int, nargs="+", default=[64, 128, 256])
    s.add_argument("--widths", type=int, nargs="+", default=[256, 512])
    s.set_defaults(func=sweep)
    return p


def main(argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    if a.split == "test" and not a.final:
        raise SystemExit("refusing to score test without --final (test is never used to choose anything)")
    a.func(a)


if __name__ == "__main__":
    main()
