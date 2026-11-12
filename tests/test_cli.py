import pytest

from flowgate.cli import main


def test_test_split_needs_final_flag():
    with pytest.raises(SystemExit, match="--final"):
        main(["r1", "--split", "test"])


def test_loads_cached_clinc_train_split():
    from kwa.config import data_dir

    from flowgate.cli import _split
    if not (data_dir() / "cache").exists():
        pytest.skip("no kwa embedding cache on this machine")
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
    from kwa.gates import SplitData

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
