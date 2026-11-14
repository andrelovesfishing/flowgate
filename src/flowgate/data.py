"""CLINC150 ("plus") questions -> frozen ModernBERT-base embeddings, cached as .npz.

The recipe: up to 64 tokens per question, mean of the last hidden layer over real tokens, fp32
compute, fp16 storage. Splits: train (in-scope only), train_oos (the train split's out-of-scope
questions, used only for evaluation), val and test (both mixed). Out-of-scope label = -1.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "train_oos", "val", "test")
CLINC_OOS_ID = 42
BACKBONE, DATASET = "answerdotai/ModernBERT-base", "clinc150"


def data_dir() -> Path:
    """Where caches and runs live: $FLOWGATE_DATA_DIR, else ~/kwa-data."""
    return Path(os.environ.get("FLOWGATE_DATA_DIR") or os.environ.get("KWA_DATA_DIR") or Path.home() / "kwa-data")


@dataclass
class SplitData:
    labels: np.ndarray
    emb: dict[str, np.ndarray] = field(default_factory=dict)


def split_labels(raw: np.ndarray, names: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    """(positions kept, labels): in-scope intents renumbered 0..149, out-of-scope = -1."""
    if names[CLINC_OOS_ID] != "oos":
        raise RuntimeError(f"expected label {CLINC_OOS_ID} to be 'oos', got {names[CLINC_OOS_ID]!r}")
    mapping = np.full(len(names), -1, dtype=np.int64)
    mapping[[i for i in range(len(names)) if i != CLINC_OOS_ID]] = np.arange(len(names) - 1)
    labels = mapping[np.asarray(raw)]
    keep = labels >= 0 if split == "train" else labels < 0 if split == "train_oos" else np.ones(len(labels), bool)
    pos = np.flatnonzero(keep)
    return pos, labels[pos]


@functools.lru_cache(maxsize=None)
def _clinc_raw(hf_split: str):
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus", split=hf_split)
    col = "intent" if "intent" in ds.column_names else "label"
    return list(ds["text"]), np.asarray(ds[col], dtype=np.int64), list(ds.features[col].names)


def load_clinc(split: str) -> tuple[list[str], np.ndarray]:
    hf_split = {"train": "train", "train_oos": "train", "val": "validation", "test": "test"}[split]
    texts, raw, names = _clinc_raw(hf_split)
    pos, labels = split_labels(raw, names, split)
    return [texts[i] for i in pos], labels


def load_backbone(name: str = BACKBONE, device: str = "cpu"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(name)
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    model = AutoModel.from_pretrained(name, config=cfg, attn_implementation="eager")
    return model.float().to(device).eval(), AutoTokenizer.from_pretrained(name)


@torch.no_grad()
def embed_texts(texts: list[str], model, tokenizer, max_len: int = 64, batch_size: int = 128,
                device: str = "cpu") -> np.ndarray:
    ids = tokenizer(list(texts), truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
    out = []
    for b in range(0, len(ids), batch_size):
        rows = ids[b : b + batch_size]
        width = max(len(r) for r in rows)
        x = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            x[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        h = model(input_ids=x.to(device), attention_mask=mask.to(device)).last_hidden_state
        m = mask.to(device).unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1).clamp_min(1.0)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def cache_path(root: Path | None = None) -> Path:
    return Path(root or data_dir()) / "cache" / BACKBONE.replace("/", "__") / DATASET


def build_cache(device: str = "cpu", max_len: int = 64, root: Path | None = None) -> Path:
    """Embed every split once (downloads the dataset and model on first use)."""
    path = cache_path(root)
    path.mkdir(parents=True, exist_ok=True)
    model, tok = load_backbone(device=device)
    for split in SPLITS:
        if (path / f"{split}.npz").exists():
            print(f"  {split}: cached", flush=True)
            continue
        texts, labels = load_clinc(split)
        print(f"  {split}: embedding {len(texts)} questions...", flush=True)
        emb = embed_texts(texts, model, tok, max_len, device=device)
        np.savez(path / f"{split}.npz", label=labels, emb=emb.astype(np.float16), max_len=np.int64(max_len))
    return path


def load_cache(split: str, root: Path | None = None) -> dict[str, np.ndarray]:
    with np.load(cache_path(root) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}
