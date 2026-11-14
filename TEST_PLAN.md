# Test plan (fixed before the test split is used)

Everything below was chosen on `cal` (val + train_oos). The test split has not been scored yet. It is scored once, with exactly these commands, and every result is reported, including claims that fail.

## What goes to test
- **Teacher:** `pca128-w512-L4-st10000-lr0.001-s0`, the exact 32-step ODE likelihood ratio, plus its 2-step version (the cheapest teacher that matched on cal).
- **Main method:** the flow-map student trained on in-scope paths only (`student-flowmap-train`), at **2 jumps**.
- **Comparisons:** the student trained with off-manifold starts (`student-flowmap-all`), the one-shot regressor (`student-direct-all`), GMM-150 ratio, top-1 cosine, Gaussian.
- Nothing else is scored on test.

## Claims and what counts as success
Paired stratified bootstrap, 2000 resamples, 95% intervals.

| # | Claim | Success means |
|---|---|---|
| C1 | The flow teacher beats the GMM | teacher-k32 − GMM: lower bound > 0 |
| C2 | The 2-jump student keeps the teacher's accuracy | student-k2 − teacher-k32: whole interval inside ±0.01 |
| C3 | The 2-jump student beats the GMM | student-k2 − GMM: point estimate > 0 (interval reported) |
| C4 | Learning the paths is what makes the copy work | flowmap-all-k1 − direct-all-k1 (same data and budget): lower bound > 0 |
| C5 | Speed | ms per query from the `flowgate time` pass (batch 1, 1 thread, from the raw embedding): student-k2 vs teacher-k2 and teacher-k32 |

Reported without a success rule: the in-scope-only vs all-starts difference, the other jump counts, and per-point errors against the teacher.

## Commands (in order; a crash may be rerun, nothing else)
```
uv run python -u -m flowgate.cli time
uv run python -u -m flowgate.cli r1 --d 128 --width 512 --split test --final
uv run python -u -m flowgate.cli r1 --d 128 --width 512 --ode-steps 2 --split test --final
uv run python -u -m flowgate.cli r2 --starts train --split test --final
uv run python -u -m flowgate.cli r2 --split test --final
uv run python -u -m flowgate.cli r2 --mode direct --split test --final
uv run python -u -m flowgate.cli report --split test --final
```

## Known limits (stated up front)
- Hyperparameters (d, width) were chosen on `cal`, which contains some labelled out-of-scope examples. Training never sees out-of-scope data.
- One dataset (CLINC150) and one encoder (ModernBERT-base). Seed stability was checked on `cal` only: three teachers scored 0.894, 0.897 and 0.896.
