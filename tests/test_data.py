import numpy as np
import pytest

from flowgate.data import CLINC_OOS_ID, cache_path, load_cache, split_labels


def _names():
    names = [f"intent{i}" for i in range(151)]
    names[CLINC_OOS_ID] = "oos"
    return names


def test_oos_becomes_minus_one_and_in_scope_ids_close_the_gap():
    raw = np.array([0, CLINC_OOS_ID, 41, 43, 150])
    keep, labels = split_labels(raw, _names(), "val")
    assert keep.tolist() == [0, 1, 2, 3, 4] and labels.tolist() == [0, -1, 41, 42, 149]


def test_train_keeps_in_scope_only_and_train_oos_keeps_oos_only():
    raw = np.array([0, CLINC_OOS_ID, 5, CLINC_OOS_ID])
    k_tr, l_tr = split_labels(raw, _names(), "train")
    k_oos, l_oos = split_labels(raw, _names(), "train_oos")
    assert k_tr.tolist() == [0, 2] and (l_tr >= 0).all()
    assert k_oos.tolist() == [1, 3] and (l_oos == -1).all()


def test_rejects_a_label_set_where_oos_moved():
    names = _names()
    names[CLINC_OOS_ID], names[0] = "intent0", "oos"
    with pytest.raises(RuntimeError, match="oos"):
        split_labels(np.array([0]), names, "val")


def test_load_cache_reads_the_npz_layout(tmp_path):
    p = cache_path(tmp_path)
    p.mkdir(parents=True)
    np.savez(p / "val.npz", label=np.array([3, -1]), emb=np.ones((2, 4), np.float16))
    z = load_cache("val", tmp_path)
    assert z["label"].tolist() == [3, -1] and z["emb"].shape == (2, 4)


def test_embedding_recipe_reproduces_the_cached_embeddings():
    """The embed step here must give the same vectors as the cache every result was computed from."""
    if not (cache_path() / "val.npz").exists():
        pytest.skip("no embedding cache on this machine")
    from flowgate.data import embed_texts, load_backbone, load_clinc
    texts, labels = load_clinc("val")
    model, tok = load_backbone()
    got = embed_texts(texts[:8], model, tok)
    ref = load_cache("val")
    assert np.array_equal(labels[:8], ref["label"][:8])
    np.testing.assert_allclose(got.astype(np.float16).astype(np.float32), ref["emb"][:8].astype(np.float32),
                               rtol=0, atol=2e-3)
