"""Regenerate results/ from the saved runs: claims table, frontier chart, and copies of the reports.

    uv run python scripts/make_results.py

Everything in results/ comes from this script; never edit it by hand.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from flowgate.cli import TEACHER, root  # noqa: E402
from flowgate.stats import auroc_oos, paired_auroc_diff  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MAIN, GMM = "student-flowmap-train-k2", "gmm150_ratio_pca128"
CLAIMS = [  # fixed in TEST_PLAN.md before the test split was scored
    ("C1", "The flow teacher beats the GMM", "teacher-k32", GMM, "lower bound > 0", lambda d, lo, hi: lo > 0),
    ("C2", "The 2-jump student keeps the teacher's accuracy", MAIN, "teacher-k32", "interval inside ±0.01",
     lambda d, lo, hi: -0.01 < lo and hi < 0.01),
    ("C3", "The 2-jump student beats the GMM", MAIN, GMM, "point estimate > 0", lambda d, lo, hi: d > 0),
    ("C4", "Learning the paths matters (flow map vs one-shot regressor, same data)", "student-flowmap-all-k1",
     "student-direct-all-k1", "lower bound > 0", lambda d, lo, hi: lo > 0),
]
FONTS = Path(__file__).resolve().parent / "fonts"  # Instrument Sans, SIL OFL (see fonts/OFL.txt)
# one hue family per theme: ink = all text, ours = the flow map, teacher, ctx = baselines,
# top/bottom = background wash, bloom = soft accent glow
THEMES = {
    "light": dict(ink="#0E2523", ours="#1F7A6C", teacher="#163F3B", ctx="#8EAEA8",
                  top="#F2F6F3", bottom="#DCE7E2", bloom="#F2E2A0"),
    "dark": dict(ink="#F1F2DF", ours="#D4E04A", teacher="#9DA37A", ctx="#6B7058",
                 top="#262A1C", bottom="#14160E", bloom="#4A5326"),
}


def load_scores(split: str):
    scores, labels = {}, None
    for f in sorted((root() / "runs").glob(f"r[12]-{TEACHER}-*-{split}/scores.npz")):
        with np.load(f) as z:
            labels = z["labels"]
            for k in z.files:
                if k.startswith("flowratio"):  # r1 runs store the teacher under the gate's name
                    scores[f"teacher-k{re.search(r'-ode(\d+)-', f.parent.name).group(1)}"] = z[k]
                elif k != "labels":
                    scores[k] = z[k]
    return scores, labels < 0


def claims_table(scores, is_oos, ms) -> str:
    rows = ["| # | Claim | Difference in AUROC [95% CI] | Success rule | Result |", "|---|---|---|---|---|"]
    for cid, text, a, b, rule, ok in CLAIMS:
        p = paired_auroc_diff(scores[a], scores[b], is_oos)
        rows.append(f"| {cid} | {text} | {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | {rule} | "
                    f"{'**pass**' if ok(p['diff'], p['lo'], p['hi']) else '**fail**'} |")
    rows.append(f"| C5 | Speed (ms per query, 1 CPU thread) | student {ms[MAIN]:.2f} ms vs teacher "
                f"{ms['teacher-k2']:.1f} ms (2 steps) and {ms['teacher-k32']:.0f} ms (32 steps) | reported | "
                f"{ms['teacher-k2'] / ms[MAIN]:.0f}× and {ms['teacher-k32'] / ms[MAIN]:.0f}× faster |")
    return "\n".join(rows)


def results_table(scores, is_oos, ms) -> str:
    show = [("Gaussian (one bell curve)", "gaussian"), ("Top-1 cosine (RAG default)", "top1_cosine"),
            ("GMM-150 ratio (strongest label-free baseline)", GMM),
            ("One-shot regressor (trained with off-manifold starts)", "student-direct-all-k1"),
            ("Flow teacher, 2 ODE steps", "teacher-k2"), ("Flow teacher, 32 ODE steps (exact)", "teacher-k32"),
            ("**Flow-map student, 1 jump**", "student-flowmap-train-k1"),
            ("**Flow-map student, 2 jumps (main)**", MAIN)]
    rows = ["| Method | Test AUROC | ms per query |", "|---|---|---|"]
    for label, k in show:
        rows.append(f"| {label} | {auroc_oos(scores[k], is_oos):.3f} | {f'{ms[k]:.2f}' if k in ms else '–'} |")
    return "\n".join(rows)


def _font() -> str:
    files = sorted(FONTS.glob("InstrumentSans-*.ttf"))
    for f in files:
        font_manager.fontManager.addfont(str(f))
    return "Instrument Sans" if files else "DejaVu Sans"


def chart(scores, is_oos, ms, path: Path, theme: str) -> None:
    """Accuracy vs cost: one point per method, labelled once; the flow map is the strong colour."""
    P = THEMES[theme]
    au = {k: auroc_oos(v, is_oos) for k, v in scores.items()}
    plt.rcParams.update({"font.family": _font(), "font.size": 9.5, "text.color": P["ink"],
                         "axes.labelcolor": P["ink"], "xtick.color": P["ink"], "ytick.color": P["ink"]})
    fig = plt.figure(figsize=(7.2, 4.5), dpi=200)

    # background: vertical wash within the hue family plus a soft accent glow low on the right
    h, w = 450, 720
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx / w, 1 - yy / h
    img = np.array(to_rgb(P["top"])) * y[..., None] + np.array(to_rgb(P["bottom"])) * (1 - y[..., None])
    t = np.clip(1 - np.hypot((x - 0.68) * 0.9, (y + 0.05) * 1.3) / 0.85, 0, 1) ** 1.5 * 0.9
    img = img * (1 - t[..., None]) + np.array(to_rgb(P["bloom"])) * t[..., None]
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg.imshow(img, aspect="auto", interpolation="bicubic")
    bg.axis("off")

    ax = fig.add_axes([0.11, 0.14, 0.85, 0.82])
    ax.patch.set_alpha(0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(P["ink"])
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=0, pad=6)
    ax.tick_params(axis="x", which="minor", length=0)
    ring = {"markeredgecolor": P["top"], "markeredgewidth": 1.4}
    lab = {"textcoords": "offset points", "fontsize": 9}

    for k, txt in [(GMM, "GMM-150"), ("student-direct-all-k1", "one-shot regressor"), ("top1_cosine", "top-1 cosine")]:
        ax.plot([ms[k]], [au[k]], "o", color=P["ctx"], markersize=8, **ring)
        ax.annotate(txt, (ms[k], au[k]), xytext=(9, -3.5), **lab)
    te = ["teacher-k2", "teacher-k32"]
    ax.plot([ms[k] for k in te], [au[k] for k in te], "-o", color=P["teacher"], lw=1.6, markersize=7, **ring)
    ax.annotate("exact likelihood", (np.sqrt(ms[te[0]] * ms[te[1]]), (au[te[0]] + au[te[1]]) / 2), xytext=(0, -16),
                ha="center", **lab)
    st = [f"student-flowmap-train-k{k}" for k in (1, 2, 4, 8)]
    ax.plot([ms[k] for k in st], [au[k] for k in st], "-o", color=P["ours"], lw=2.2, markersize=9, **ring)
    ax.annotate("flow map", (ms[MAIN], au[MAIN]), xytext=(0, 10), ha="center", fontweight=700, **lab)

    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    ax.set_xlabel("ms per query", labelpad=6)
    ax.set_ylabel("AUROC", labelpad=6)
    ax.set_yticks([0.84, 0.86, 0.88, 0.90])
    ax.set_ylim(0.825, 0.915)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rep = root() / "reports"
    shutil.copy(rep / f"timing-{TEACHER}.json", OUT / "timing.json")
    for split in ("cal", "test"):
        shutil.copy(rep / f"frontier-{TEACHER}-{split}.json", OUT / f"frontier-{split}.json")
    ms = json.loads((OUT / "timing.json").read_text())["ms"]
    scores, is_oos = load_scores("test")
    md = (f"# Test results (generated by scripts/make_results.py)\n\nTest split: {len(is_oos)} questions, "
          f"{int(is_oos.sum())} out of scope. Paired stratified bootstrap, 2000 resamples.\n\n"
          f"## Pre-registered claims (TEST_PLAN.md)\n\n{claims_table(scores, is_oos, ms)}\n\n"
          f"## Main results\n\n{results_table(scores, is_oos, ms)}\n")
    (OUT / "test.md").write_text(md, encoding="utf-8")
    for theme in THEMES:
        chart(scores, is_oos, ms, OUT / f"frontier-{theme}.png", theme)
    print(md)


if __name__ == "__main__":
    main()
