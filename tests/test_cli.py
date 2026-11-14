import pytest

from flowgate.cli import main


def test_test_split_needs_final_flag():
    with pytest.raises(SystemExit, match="--final"):
        main(["r1", "--split", "test"])


def test_loads_cached_clinc_train_split():
    from flowgate.cli import _split
    from flowgate.data import cache_path
    if not (cache_path() / "train.npz").exists():
        pytest.skip("no embedding cache on this machine")
    tr = _split("train")
    assert tr.emb["frozen"].shape == (15000, 768) and (tr.labels >= 0).all()


def test_model_key_depends_on_training_settings_only():
    from flowgate.cli import _parser, model_key
    p = _parser()
    a = p.parse_args(["r1", "--d", "64"])
    b = p.parse_args(["r1", "--d", "64", "--ode-steps", "128", "--subsample", "500", "--time-probe"])
    c = p.parse_args(["r1", "--d", "64", "--width", "256"])
    assert model_key(a) == model_key(b) != model_key(c)


def test_baselines_return_per_point_scores_and_cache_them(tmp_path):
    import numpy as np
    from flowgate.data import SplitData

    from flowgate.cli import baselines
    rng = np.random.default_rng(0)
    tr = SplitData(labels=np.zeros(300, int), emb={"frozen": rng.normal(size=(300, 8)).astype(np.float32)})
    ev = SplitData(labels=np.r_[np.zeros(40, int), -np.ones(10, int)],
                   emb={"frozen": rng.normal(size=(50, 8)).astype(np.float32)})
    cache = tmp_path / "b.npz"
    s = baselines(tr, ev, d=4, kind="pca", gmm=False, cache=cache)
    assert set(s) == {"gaussian", "top1_cosine"} and all(v.shape == (50,) for v in s.values())
    again = baselines(tr, ev, d=4, kind="pca", gmm=False, cache=cache)
    assert all(np.array_equal(s[k], again[k]) for k in s)


def test_r2_and_report_on_test_split_need_final_flag():
    for cmd in (["r2", "--split", "test"], ["report", "--split", "test"]):
        with pytest.raises(SystemExit, match="--final"):
            main(cmd)


def test_student_key_depends_on_training_settings_only():
    from flowgate.cli import _parser, student_key
    p = _parser()
    a = p.parse_args(["r2"])
    b = p.parse_args(["r2", "--ks", "1", "--latency-n", "5"])
    c = p.parse_args(["r2", "--mode", "direct"])
    d = p.parse_args(["r2", "--starts", "train"])
    assert student_key(a) == student_key(b) and len({student_key(x) for x in (a, c, d)}) == 3


def test_smoke_runs_get_their_own_keys():
    from flowgate.cli import _apply_smoke, _parser, student_key, traj_key
    p = _parser()
    a, s = p.parse_args(["r2"]), _apply_smoke(p.parse_args(["r2", "--smoke"]))
    assert student_key(a) != student_key(s) and traj_key(a) != traj_key(s) and s.steps < a.steps


def test_compare_gives_auroc_paired_cis_and_split_errors_vs_teacher():
    import numpy as np

    from flowgate.cli import compare
    rng = np.random.default_rng(0)
    is_oos = np.r_[np.zeros(300, bool), np.ones(100, bool)]
    teacher = rng.normal(size=400) - 2.0 * is_oos
    student = teacher + np.where(is_oos, 0.5, 0.1)  # off by 0.1 in-scope, 0.5 OOS
    scores = {"teacher-k32": teacher, "student-flowmap-all-k1": student, "gmm150_ratio_pca8": rng.normal(size=400)}
    rows = {r["method"]: r for r in compare(scores, is_oos, gmm="gmm150_ratio_pca8", teacher="teacher-k32",
                                            costs={"student-flowmap-all-k1": {"calls": 1, "ms": 0.1}})}
    s = rows["student-flowmap-all-k1"]
    assert s["calls"] == 1 and "vs_gmm" in s and "vs_teacher" in s
    assert abs(s["err_in"] - 0.1) < 1e-9 and abs(s["err_oos"] - 0.5) < 1e-9
    assert "vs_gmm" not in rows["gmm150_ratio_pca8"] and "vs_teacher" not in rows["teacher-k32"]
    assert "err_in" not in rows["gmm150_ratio_pca8"]


def test_student_method_names_carry_the_seed_only_when_nonzero():
    from flowgate.cli import method_prefix
    assert method_prefix("flowmap", "all", 0) == "student-flowmap-all"  # seed-0 names stay as already saved
    assert method_prefix("direct", "train", 2) == "student-direct-train-s2"


def test_timing_pass_overrides_per_run_ms_and_adds_baselines():
    from flowgate.cli import apply_timing
    costs = {"teacher-k2": {"calls": 260, "ms": 37.0}, "student-x-k1": {"calls": 1, "ms": 0.95}}
    out = apply_timing(costs, {"teacher-k2": 11.0, "gmm": 0.2, "unused": 5.0}, ["teacher-k2", "student-x-k1", "gmm", "top1"])
    assert out["teacher-k2"] == {"calls": 260, "ms": 11.0, "ms_src": "timing"}
    assert out["student-x-k1"] == {"calls": 1, "ms": 0.95, "ms_src": "run"}
    assert out["gmm"] == {"ms": 0.2, "ms_src": "timing"} and "top1" not in out and "unused" not in out


def test_timed_functions_compute_the_same_scores_as_the_scored_methods():
    """Every timer starts from the raw embedding and must return the very score that method is
    evaluated on, so the timing pass times the real thing."""
    import numpy as np
    import torch
    from flowgate.data import SplitData
    from sklearn.mixture import GaussianMixture

    from flowgate.cli import baselines, gate_timers
    from flowgate.distill import Student, student_score
    from flowgate.gate import FlowRatioGate
    from flowgate.gmm import TiedGMMScorer
    from flowgate.net import FlowMapNet
    from flowgate.space import Space
    from flowgate.train import TrainConfig
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(300, 6)).astype(np.float32)
    tr = SplitData(labels=np.zeros(300, int), emb={"frozen": Xtr})
    g = FlowRatioGate(d=3, width=16, depth=2, ode_steps=4,
                      train_cfg=TrainConfig(steps=20, batch=64, eval_every=10)).fit(tr)
    sp = Space.fit(Xtr, 3, "pca")
    gm = GaussianMixture(2, covariance_type="tied", random_state=0).fit(sp.transform(Xtr).astype(np.float64))
    st = Student(FlowMapNet(3, 16, 2)); st.net.load_state_dict(g.net.state_dict())
    timers = gate_timers(g, sp, TiedGMMScorer.from_sklearn(gm), Xtr,
                         {"student-flowmap-all": (st, "flowmap", [1, 2])}, ladder=(4,))
    assert set(timers) == {"gmm150_ratio_pca3", "top1_cosine", "teacher-k4",
                           "student-flowmap-all-k1", "student-flowmap-all-k2"}
    q = rng.normal(size=(1, 6)).astype(np.float32)
    ev, qt = SplitData(labels=np.zeros(1, int), emb={"frozen": q}), torch.from_numpy(q)
    assert abs(float(timers["teacher-k4"](qt)[0]) - g.score(ev)[0]) < 1e-3
    z = torch.from_numpy(g.space.transform(q))
    assert abs(float(timers["student-flowmap-all-k2"](qt)[0]) - float(student_score(st, z, 2)[0])) < 1e-3
    zg = sp.transform(q).astype(np.float64)
    assert abs(float(timers["gmm150_ratio_pca3"](qt)[0]) - (gm.score_samples(zg)[0] + 0.5 * (zg**2).sum())) < 1e-3
    want = baselines(tr, ev, d=3, kind="pca", gmm=False)["top1_cosine"][0]
    assert abs(float(timers["top1_cosine"](qt)[0]) - want) < 1e-5
