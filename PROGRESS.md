# DeepAlgPro — experiment log

Test set: `data/all.test.fasta`, n=1420 (710 allergen / 710 non-allergen).
Single-measurement 95% CI is +/- 1.44 pp, so differences under ~1.5 pp are not
resolvable from one run. Fold-to-fold sd is the honest error bar.

## Headline results

| # | encoder | arch | Acc | F1 | AUROC | params | protocol |
|---|---------|------|-----|----|-------|--------|----------|
| 0 | one-hot | base | 0.91620 | 0.91506 | 0.96305 | 46,145 | published (10 folds, `model.pt`) |
| 1 | one-hot | base | **0.9014 +/- 0.0062** | 0.9013 +/- 0.0073 | 0.9610 +/- 0.0034 | 46,145 | 5 folds, 100 ep, b=32, patience 15 |
| 1b | one-hot | base | 0.8893 +/- 0.0115 | 0.8898 +/- 0.0082 | 0.9557 +/- 0.0036 | 46,145 | 5 folds, 100 ep, **b=72**, patience 15 |
| 2 | ProtT5  | base | 0.95915 | 0.95869 | 0.98195 | 1,073,217 | **provisional** - 1 fold, 30 ep, b=32 |

Row 2 is NOT comparable to row 1: one fold, 30 epochs (underfit). Must be rerun
at row 1's protocol before any gap is quoted.

## Row 1 - one-hot baseline, reproduced (2026-11-11)

```
python main.py -i data/all.train.fasta --mode train --tag onehot_e100 \
    --folds 5 -b 32 --epochs 100 --patience 15
python main.py -i data/all.test.fasta  --mode test  -b 64 \
    --weights weights/onehot_e100.$k.pt        # k = 1..5
```

Per-fold test metrics:

| fold | Acc | F1 | Precision | Recall | AUROC |
|------|-----|----|-----------|--------|-------|
| 1 | 0.90141 | 0.90398 | 0.88102 | 0.92817 | 0.96353 |
| 2 | 0.90070 | 0.89834 | 0.92024 | 0.87746 | 0.95641 |
| 3 | 0.90775 | 0.90922 | 0.89495 | 0.92394 | 0.96439 |
| 4 | 0.90563 | 0.90442 | 0.91618 | 0.89296 | 0.96251 |
| 5 | 0.89155 | 0.89078 | 0.89714 | 0.88451 | 0.95806 |
| **mean** | **0.90141** | **0.90135** | 0.90191 | 0.90141 | 0.96098 |
| **sd** | 0.00624 | 0.00729 | 0.01558 | 0.02293 | 0.00341 |

Checkpoints: `weights/onehot_e100.{1..5}.pt`; best fold (3) also as
`weights/onehot_e100.best.pt`. Report the MEAN, not the best fold - 0.90775 is
the max of 5 draws and quoting it as "the baseline" is selection bias.

### Findings

1. **Epoch count was the earlier failure, not the code.** An identical 5-fold run
   at 30 epochs gave 0.86972 +/- 0.00770, i.e. 4.6 pp low. Per-epoch logs showed
   best F1 at epoch 27-29 in all 5 folds - still climbing at the cap. At 100
   epochs with patience 15, folds stopped early (fold 5 at epoch 65, best at 50),
   confirming 100 is sufficient and 200 is unnecessary.
2. **Validation overstates test by ~1.5 pp.** Validation mean was 0.91618 vs test
   0.90141. The checkpoint is selected on the validation fold (best F1), so that
   number is optimistically biased by the selection itself. The validation figure
   coincidentally matching the published 0.91620 is a coincidence, not a
   reproduction. Neither DeepAlgPro nor PreAlgPro discusses this.
3. **1.48 pp short of published on accuracy, but AUROC matches** (0.9610 vs
   0.96305). Ranking quality is reproduced; the threshold-dependent metrics are
   not. Remaining protocol deviations: b=32 vs the repo default b=72, and 5 of 10
   folds.
4. **Batch size settled: b=32 wins.** b=72 (the repo default) gives
   0.8893 +/- 0.0115, i.e. 1.21 pp WORSE with ~2x the fold spread, and it is
   worse in validation too (0.9075 vs 0.9162) - a real effect, not test noise.
   So batch size does not explain the gap to 0.91620. Both experiment rows use
   b=32. Weights kept as `weights/onehot_b72.{1..5}.pt`; this doubles as a
   batch-size sensitivity ablation.
5. **The gap to published is most likely selection, not training.** `model.pt`
   is a single checkpoint, almost certainly the best of 10 folds. Best of our 5
   at b=32 was 0.90775. A best-of-10 vs mean-of-5 comparison produces exactly
   this ~1.5 pp offset, so the baseline is reproduced once compared like with
   like - which the paper does not permit, as it reports no spread.

## Rules for this experiment

- Row 2 must use row 1's exact flags. Only `conv1`'s input width changes
  (21 -> 1024); maxpool, attention and `fc1` stay byte-identical because `conv1`
  always emits 64 channels. That isolation is the whole point - it is the
  ablation PreAlgPro omitted.
- Keep all per-fold numbers: they enable a paired test between rows.
- `/content` is not storage. Download or mirror to Drive after every gate.

## Next

- [x] Batch size decided: **b=32** (b=72 was 1.21 pp worse). 2026-11-11.
- [ ] Speed fix: RAM-cache `EmbDataset` (3.63 GB unpadded fp16 fits Colab's
      12.7 GB), `num_workers=0`; then 2-epoch probe to re-time the run.
- [ ] Row 2 rerun: ProtT5, 5 folds, 100 ep, patience 15, b=32,
      `--in-channels 1024 --embeddings emb/prott5/train` (~2.5-5.5 h).
- [ ] Add `sentencepiece` to `requirements.txt`.
- [ ] Phase 3 (novelty): 3a pool positions not filters; 3b global pool +
      `Linear(64,1)`; 3c padding mask; 3d length-bucketed batching; 3e Conformer.
      Plus a one-hot x fixed-arch control row.
