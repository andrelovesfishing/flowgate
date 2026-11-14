"""flowgate CLI.

  r1     train (or reuse) one flow-ratio gate and score it on cal (val + train_oos) or, once, on test
  sweep  run r1 over a grid of d x width on cal and print one table
  r2     distil a trained r1 teacher into a few-step flow map (or the direct ablation) and score it
  report the accuracy-vs-cost frontier for one teacher, from saved runs only
  time   time every method in one quiet, interleaved pass; report uses these ms

Trained models are cached under $KWA_DATA_DIR/flowgate/models/, keyed by training settings only,
so rescoring (other ODE steps, other eval sets) never retrains. Baselines are cached per space.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from kwa.backbones import load_cache
from kwa.config import data_dir
from kwa.gates import SplitData
from kwa.metrics import auroc_oos

from .distill import (KINDS, DistillConfig, Student, Trajectories, make_starts, record, student_score,
                      train_student, val_error)
from .gate import FlowRatioGate, early_stop_split
from .gmm import TiedGMMScorer
from .latency import ms_per_query, time_interleaved
from .likelihood import ode_logp, std_normal_logpdf
from .net import FlowMapNet
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
        print(f"  fitting GMM-150 baseline in {kind}{d} (one-off, cached)...", flush=True)
        sp = Space.fit(Xtr, d, kind)
        ztr, ze = sp.transform(Xtr).astype(np.float64), sp.transform(Xe).astype(np.float64)
        gm = _gmm150().fit(ztr)
        out[f"gmm150_ratio_{kind}{d}"] = gm.score_samples(ze) + 0.5 * (ze**2).sum(1)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


def _gmm150():
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(150, covariance_type="tied", random_state=0, max_iter=200, reg_covar=1e-3)


def gmm_model(train: SplitData, d: int, kind: str) -> tuple[Space, TiedGMMScorer]:
    """The GMM-150 baseline as a model rather than scores (baselines caches only scores): same space,
    same fit settings, so the same model. Its parameters are cached."""
    Xtr = train.emb["frozen"].astype(np.float64)
    sp = Space.fit(Xtr, d, kind)
    path = root() / "baselines" / f"gmm150-params-{kind}{d}.npz"
    if path.exists():
        with np.load(path) as z:
            return sp, TiedGMMScorer.from_params({k: z[k] for k in z.files})
    print(f"  fitting GMM-150 in {kind}{d} to time it (one-off, cached)...", flush=True)
    s = TiedGMMScorer.from_sklearn(_gmm150().fit(sp.transform(Xtr).astype(np.float64)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **s.params())
    return sp, s


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


# ---------------------------------------------------------------- R2: few-step flow-map student

TEACHER = "pca128-w512-L4-st10000-lr0.001-s0"
LADDER = (1, 2, 4, 8, 16, 32)  # teacher ODE steps timed for the frontier


def _apply_smoke(a):
    if a.smoke:
        a.n_train, a.n_off, a.steps, a.eval_every = 1000, 300, 300, 100
    return a


def traj_key(a) -> str:
    return f"{a.teacher}-n{a.n_train or 'all'}-off{a.n_off}-T{a.traj_steps}"


def student_key(a) -> str:
    return f"{traj_key(a)}--{a.mode}-{a.starts}-st{a.steps}-ev{a.eval_every}-lr{a.lr:g}-s{a.seed}"


def teacher_cost(k: int, d: int, ms: float | None = None) -> dict:
    """k midpoint steps = 2k velocity calls + k exact traces of d vector-Jacobian products each."""
    return {"calls": 2 * k + k * d, **({"ms": ms} if ms is not None else {})}


def method_prefix(mode: str, starts: str, seed: int) -> str:
    return f"student-{mode}-{starts}" + (f"-s{seed}" if seed else "")


def apply_timing(costs: dict, timing_ms: dict, methods) -> dict:
    """Costs for each method, with ms from the dedicated timing pass where it has them ("timing"),
    else the ms timed inside the training run ("run", noisier). Methods with no cost are left out."""
    out = {}
    for m in methods:
        c = dict(costs.get(m, {}))
        if m in timing_ms:
            c.update(ms=timing_ms[m], ms_src="timing")
        elif "ms" in c:
            c["ms_src"] = "run"
        if c:
            out[m] = c
    return out


def get_trajectories(a, teacher: FlowRatioGate, train: SplitData) -> Trajectories:
    path = root() / "traj" / f"{traj_key(a)}.npz"
    if path.exists():
        print(f"reusing teacher paths {path.name}", flush=True)
        return Trajectories.load(path)
    z = teacher.space.transform(np.asarray(train.emb[teacher.space_key]))
    _, es = early_stop_split(len(z), teacher.train_cfg.seed, teacher.early_stop_frac)
    held = np.zeros(len(z), bool)
    held[es.numpy()] = True
    if a.n_train:
        sel = np.sort(np.random.default_rng(0).choice(len(z), a.n_train, replace=False))
        z, held = z[sel], held[sel]
    X, kind, hold = make_starts(z, np.flatnonzero(held), a.n_off, seed=0)
    print(f"recording exact teacher paths for {len(X)} starts ({a.traj_steps} steps; one-off, cached)...", flush=True)
    t0 = time.perf_counter()
    xs, L = record(teacher.net.velocity, X, steps=a.traj_steps, log=lambda s: print(s, flush=True))
    per = (time.perf_counter() - t0) / len(X)
    print(f"  {per * 1e3:.1f} ms/path; a full run (15000 + 3 x 5000 paths) would take ~{per * 30000 / 60:.0f} min",
          flush=True)
    tj = Trajectories(xs, L, kind, hold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tj.save(path)
    return tj


def _load_student(path: Path, teacher: FlowRatioGate):
    ck = torch.load(path, weights_only=True)
    st = Student(FlowMapNet(teacher.d, teacher.width, teacher.depth))
    st.load_state_dict(ck["state"])
    st.eval()
    return st, ck


def get_student(a, teacher: FlowRatioGate, tj: Trajectories):
    path = root() / "students" / f"{student_key(a)}.pt"
    if path.exists() and not a.retrain:
        print(f"reusing student {path.name}", flush=True)
        st, ck = _load_student(path, teacher)
        return st, ck["history"], ck["fit_seconds"]
    net = FlowMapNet(teacher.d, teacher.width, teacher.depth)
    net.load_state_dict(teacher.net.state_dict())  # the student starts as the teacher
    st = Student(net)
    sub = tj.subset(tj.kind == 0) if a.starts == "train" else tj
    cfg = DistillConfig(steps=a.steps, lr=a.lr, seed=a.seed, mode=a.mode, starts=a.starts, eval_every=a.eval_every)
    print(f"distilling a {a.mode} student on {int((~sub.holdout).sum())} paths ({a.starts} starts), "
          f"early-stopping on {int(sub.holdout.sum())} held-out paths...", flush=True)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    st, hist = train_student(st, sub.subset(~sub.holdout), sub.subset(sub.holdout), cfg,
                             log=lambda s: print(s, flush=True))
    fit = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": st.state_dict(), "cfg": cfg.to_dict(), "history": hist, "fit_seconds": fit}, path)
    return st, hist, fit


def compare(scores: dict, is_oos: np.ndarray, gmm: str, teacher: str, costs: dict, n_boot: int = 2000) -> list[dict]:
    """One row per method: AUROC, paired CI vs the GMM and (for flow methods) vs the 32-step teacher,
    and mean |score - teacher score| on in-scope and OOS points separately. Sorted by cost."""
    rows = []
    for name, s in scores.items():
        r = {"method": name, "auroc": auroc_oos(s, is_oos), **costs.get(name, {})}
        if name != gmm and gmm in scores:
            p = paired_auroc_diff(s, scores[gmm], is_oos, n_boot)
            r["vs_gmm"] = [p["diff"], p["lo"], p["hi"]]
        if name.startswith(("student", "teacher")) and name != teacher and teacher in scores:
            p = paired_auroc_diff(s, scores[teacher], is_oos, n_boot)
            r["vs_teacher"] = [p["diff"], p["lo"], p["hi"]]
            e = np.abs(s - scores[teacher])
            r["err_in"], r["err_oos"] = float(e[~is_oos].mean()), float(e[is_oos].mean())
        rows.append(r)
    return sorted(rows, key=lambda r: (r.get("calls", 0), -r["auroc"]))


def print_rows(rows: list[dict], title: str) -> None:
    def ci(v):
        return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if v else "-"
    print(f"\n{title}")
    print(f"  {'method':28s} {'calls/q':>8} {'ms/q':>8} {'AUROC':>6}  {'vs GMM [95% CI]':>24}  "
          f"{'vs teacher-k32 [95% CI]':>24}  {'|err| in':>8} {'|err| OOS':>9}")
    for r in rows:
        calls = str(r["calls"]) if "calls" in r else "-"
        ms = (f"{r['ms']:.2f}" + ("*" if r.get("ms_src") == "run" else "")) if "ms" in r else "-"
        err = f"{r['err_in']:8.2f} {r['err_oos']:9.2f}" if "err_in" in r else f"{'-':>8} {'-':>9}"
        print(f"  {r['method']:28s} {calls:>8} {ms:>8} {r['auroc']:6.3f}  {ci(r.get('vs_gmm')):>24}  "
              f"{ci(r.get('vs_teacher')):>24}  {err}")
    print("  calls/q = network calls per query (forward passes + vector-Jacobian products); ms/q = batch 1, "
          "1 thread, median.\n  |err| = mean |score - 32-step teacher score| in nats. Ceiling: labelled relmaha 0.950 on cal.")
    if any(r.get("ms_src") == "run" for r in rows):
        print("  * ms timed inside a training run (noisy); unmarked ms are from the `flowgate time` pass.")


def _load_teacher(key: str) -> FlowRatioGate:
    path = root() / "models" / f"{key}.pt"
    if not path.exists():
        raise SystemExit(f"no trained teacher {path.name}; train it with flowgate r1 first")
    return FlowRatioGate.load(path)


def _teacher_scores(a, teacher: FlowRatioGate, ev: SplitData) -> np.ndarray:
    f = root() / "runs" / f"r1-{a.teacher}-ode32-{a.split}" / "scores.npz"
    if f.exists():
        with np.load(f) as z:
            return z[teacher.name]
    print("no saved 32-step teacher scores for this split; scoring now (exact ODE)...", flush=True)
    teacher.ode_steps = 32
    return teacher.score(ev)


def r2(a: argparse.Namespace) -> dict:
    _apply_smoke(a)
    teacher = _load_teacher(a.teacher)
    train = _split("train")
    tj = get_trajectories(a, teacher, train)
    st, hist, fit = get_student(a, teacher, tj)
    st.eval()
    ks = tuple(a.ks) if a.mode == "flowmap" else (1,)
    held = {KINDS[k]: val_error(st, tj.subset(tj.holdout & (tj.kind == k)), a.mode, ks) for k in range(len(KINDS))}

    ev = eval_set(a.split, None, 0)
    is_oos = ev.labels < 0
    ze = torch.from_numpy(teacher.space.transform(ev.emb["frozen"]))
    prefix = method_prefix(a.mode, a.starts, a.seed)
    scores = {f"{prefix}-k{k}": student_score(st, ze, k, a.mode).double().numpy() for k in ks}
    print("timing (batch 1, 1 thread)...", flush=True)
    costs = {f"{prefix}-k{k}": {"calls": k, "ms": ms_per_query(lambda q, k=k: student_score(st, q, k, a.mode), ze,
                                                              n=a.latency_n)} for k in ks}
    teacher_ms = {k: ms_per_query(lambda q, k=k: ode_logp(teacher.net.velocity, q, steps=k), ze,
                                  n=max(3, a.latency_n // 4)) for k in LADDER}
    scores["teacher-k32"] = _teacher_scores(a, teacher, ev)
    costs["teacher-k32"] = teacher_cost(32, teacher.d, teacher_ms[32])
    cache = root() / "baselines" / f"{a.split}-{teacher.kind}{teacher.d}.npz"
    scores.update(baselines(train, ev, teacher.d, teacher.kind, gmm=True, cache=cache))
    gmm = f"gmm150_ratio_{teacher.kind}{teacher.d}"
    rows = compare(scores, is_oos, gmm, "teacher-k32", costs)
    print_rows(rows, f"R2 on {a.split} ({len(ev.labels)} pts, {int(is_oos.sum())} OOS): {a.mode} student, "
                     f"{a.starts} starts, trained {hist[-1]['step']} steps in {fit / 60:.1f} min")
    print("\n  held-out distillation paths, mean |score error| (nats) by start kind:")
    for kind, e in held.items():
        print(f"    {kind:7s} " + "  ".join(f"k{k} {v:.2f}" for k, v in e.items()))
    if a.smoke:
        print(f"\nsmoke: {fit / hist[-1]['step'] * 1e3:.0f} ms/step, so 10k steps would take ~"
              f"{fit / hist[-1]['step'] * 1e4 / 60:.0f} min (at this run's power/CPU state)")
    res = {"config": {k: v for k, v in vars(a).items() if k != "func"}, "student": student_key(a),
           "code_sha": _git_sha(), "history": hist, "fit_seconds": fit, "held_out_error": held,
           "teacher_ms": teacher_ms, "rows": rows}
    out = root() / "runs" / f"r2-{student_key(a)}-{a.split}{'-smoke' if a.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
    np.savez(out / "scores.npz", labels=ev.labels, **{k: v for k, v in scores.items() if k.startswith("student")})
    print(f"wrote {out / 'result.json'}\n", flush=True)
    return res


def report(a: argparse.Namespace) -> list[dict]:
    """The accuracy-vs-cost frontier for one teacher, built only from saved runs (no recomputation):
    baselines, the teacher at every saved ODE step count (r1 --ode-steps), and every r2 student."""
    teacher = _load_teacher(a.teacher)
    runs = root() / "runs"
    scores, costs, teacher_ms, labels, base = {}, {}, {}, None, {}
    for f in sorted(runs.glob(f"r2-{a.teacher}-*-{a.split}/result.json")):  # "-smoke" runs end differently
        res = json.loads(f.read_text())
        teacher_ms.update({int(k): v for k, v in res["teacher_ms"].items()})
        with np.load(f.parent / "scores.npz") as z:
            labels = z["labels"]
            scores.update({k: z[k] for k in z.files if k.startswith("student")})
        costs.update({r["method"]: {"calls": r["calls"], "ms": r["ms"]} for r in res["rows"]
                      if r["method"].startswith("student")})
    for f in sorted(runs.glob(f"r1-{a.teacher}-ode*-{a.split}/scores.npz")):
        k = int(re.search(r"-ode(\d+)-", f.parent.name).group(1))
        with np.load(f) as z:
            labels = z["labels"]
            scores[f"teacher-k{k}"] = z[teacher.name]
            base = {b: z[b] for b in z.files if b not in ("labels", teacher.name)}
        costs[f"teacher-k{k}"] = teacher_cost(k, teacher.d, teacher_ms.get(k))
    if labels is None:
        raise SystemExit(f"no saved runs for teacher {a.teacher} on {a.split}")
    scores.update(base)
    tfile = root() / "reports" / f"timing-{a.teacher}.json"
    tm = json.loads(tfile.read_text()) if tfile.exists() else {"ms": {}}
    costs = apply_timing(costs, tm["ms"], list(scores))
    rows = compare(scores, labels < 0, f"gmm150_ratio_{teacher.kind}{teacher.d}", "teacher-k32", costs)
    print_rows(rows, f"FRONTIER on {a.split}: teacher {a.teacher}")
    print(f"  timing pass: {tm['when']} ({tm['protocol']})" if "when" in tm else
          "  no timing pass yet: run `flowgate time` (machine idle, plugged in) for comparable ms")
    out = root() / "reports" / f"frontier-{a.teacher}-{a.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"wrote {out}")
    return rows


def gate_timers(teacher: FlowRatioGate, gmm_space: Space, gmm: TiedGMMScorer, train_emb: np.ndarray,
                students: dict, ladder=LADDER) -> dict:
    """One function per method, each taking raw embeddings [1, D] and returning that method's gate
    score, i.e. the gate's whole cost once the backbone has produced the embedding (projection
    included). students: {method prefix: (Student, mode, ks)}."""
    def proj(sp: Space):
        mu, A = torch.from_numpy(sp.mu).float(), torch.from_numpy(sp.A).float()
        return lambda x: (x - mu) @ A
    zt, zg = proj(teacher.space), proj(gmm_space)
    g32 = TiedGMMScorer.from_params(gmm.params(), dtype=torch.float32)
    B = torch.from_numpy(np.asarray(train_emb, dtype=np.float32))
    BT = (B / B.norm(dim=1, keepdim=True)).T.contiguous()

    def teacher_k(k):
        def f(x):
            z = zt(x)
            return ode_logp(teacher.net.velocity, z, steps=k) - std_normal_logpdf(z)
        return f
    fns = {f"gmm150_ratio_{teacher.kind}{teacher.d}": lambda x: g32(zg(x)),
           "top1_cosine": lambda x: ((x / x.norm(dim=1, keepdim=True)) @ BT).max(1).values}
    fns.update({f"teacher-k{k}": teacher_k(k) for k in ladder})
    for prefix, (st, mode, ks) in students.items():
        for k in ks:
            fns[f"{prefix}-k{k}"] = lambda x, st=st, k=k, mode=mode: student_score(st, zt(x), k, mode)
    return fns


def timing(a: argparse.Namespace) -> dict:
    """Time every method of one teacher's frontier in one process, interleaved, on an idle machine.
    `report` takes its ms from here. Queries are cal embeddings (no labels used)."""
    teacher = _load_teacher(a.teacher)
    train, ev = _split("train"), eval_set("cal", None, 0)
    gsp, gmm = gmm_model(train, teacher.d, teacher.kind)
    gname = f"gmm150_ratio_{teacher.kind}{teacher.d}"
    check, cache = None, root() / "baselines" / f"cal-{teacher.kind}{teacher.d}.npz"
    if cache.exists():  # the timed GMM must be the baseline's GMM
        with np.load(cache) as z:
            ref = z[gname]
        got = gmm(torch.from_numpy(gsp.transform(ev.emb["frozen"]).astype(np.float64))).numpy()
        check = float(np.abs(got - ref).max())
        print(f"  timed GMM vs the baseline's cached scores: max |diff| {check:.1e}"
              + ("" if check < 1e-6 else "  <-- NOT the same model; its timing is still representative"), flush=True)
    students = {}
    for f in sorted((root() / "runs").glob(f"r2-{a.teacher}-*-cal/result.json")):
        res = json.loads(f.read_text())
        c = res["config"]
        st, _ = _load_student(root() / "students" / f"{res['student']}.pt", teacher)
        students[method_prefix(c["mode"], c["starts"], c["seed"])] = (st, c["mode"],
                                                                      c["ks"] if c["mode"] == "flowmap" else [1])
    fns = gate_timers(teacher, gsp, gmm, train.emb["frozen"], students)
    X = torch.from_numpy(ev.emb["frozen"][np.random.default_rng(0).permutation(len(ev.labels))[: a.queries]])
    protocol = f"batch 1, 1 thread, {a.rounds} interleaved rounds x {a.n} queries, median"
    print(f"timing {len(fns)} methods from the raw embedding ({protocol}); keep the machine idle...", flush=True)
    t = time_interleaved(fns, X.float(), n=a.n, rounds=a.rounds)
    print(f"\n  {'method':30s} {'median ms':>10} {'p10':>8} {'p90':>8}")
    for k, v in sorted(t.items(), key=lambda kv: kv[1]["median"]):
        print(f"  {k:30s} {v['median']:10.3f} {v['p10']:8.3f} {v['p90']:8.3f}")
    res = {"when": datetime.now().isoformat(timespec="minutes"), "protocol": protocol, "teacher": a.teacher,
           "machine": f"{platform.platform()} | {platform.processor()}", "torch": torch.__version__,
           "code_sha": _git_sha(), "gmm_check_max_abs_diff": check,
           "ms": {k: v["median"] for k, v in t.items()}, "detail": t}
    out = root() / "reports" / f"timing-{a.teacher}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return res


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

    r = sub.add_parser("r2", help="distil a trained teacher into a few-step flow map and score it")
    r.add_argument("--teacher", default=TEACHER, help="r1 model key of the teacher")
    r.add_argument("--mode", choices=["flowmap", "direct"], default="flowmap")
    r.add_argument("--starts", choices=["all", "train"], default="all", help="train = in-scope paths only (ablation)")
    r.add_argument("--steps", type=int, default=10_000)
    r.add_argument("--lr", type=float, default=3e-4)
    r.add_argument("--eval-every", type=int, default=500)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-train", type=int, default=0, help="train starts to keep (0 = all 15000)")
    r.add_argument("--n-off", type=int, default=5000, help="starts per off-manifold kind")
    r.add_argument("--traj-steps", type=int, default=32)
    r.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8], help="flow-map jumps to score with")
    r.add_argument("--latency-n", type=int, default=20)
    r.add_argument("--split", choices=["cal", "test"], default="cal")
    r.add_argument("--final", action="store_true", help="required for --split test; use once")
    r.add_argument("--retrain", action="store_true", help="ignore a cached student")
    r.add_argument("--smoke", action="store_true", help="tiny run (1000 + 3 x 300 paths, 300 steps) to measure cost")
    r.set_defaults(func=r2)
    rp = sub.add_parser("report", help="accuracy-vs-cost frontier for one teacher, from saved runs")
    rp.add_argument("--teacher", default=TEACHER)
    rp.add_argument("--split", choices=["cal", "test"], default="cal")
    rp.add_argument("--final", action="store_true", help="required for --split test; use once")
    rp.set_defaults(func=report)
    tp = sub.add_parser("time", help="time every method in one quiet, interleaved pass (report reads it)")
    tp.add_argument("--teacher", default=TEACHER)
    tp.add_argument("--n", type=int, default=20, help="queries per method per round")
    tp.add_argument("--rounds", type=int, default=5)
    tp.add_argument("--queries", type=int, default=200, help="distinct cal embeddings to cycle through")
    tp.set_defaults(func=timing)
    return p


def main(argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    if getattr(a, "split", None) == "test" and not a.final:
        raise SystemExit("refusing to score test without --final (test is never used to choose anything)")
    a.func(a)


if __name__ == "__main__":
    main()
