# TriFuseRec: Adaptive Context-Sequence-Graph Fusion for Personalized Recommendation

TriFuseRec is a personalized recommendation framework that adaptively combines three complementary sources of information: user-item features, sequential interaction history, and user-item graph structure.

The project integrates three recommendation architectures:

- **DeepFM:** Captures user and item feature interactions.
- **SASRec:** Models sequential user preferences using self-attention.
- **NGCF:** Learns collaborative relationships through graph-based representation learning.

Unlike traditional recommendation systems that rely on a single model or fixed ensemble weights, TriFuseRec introduces a personalized gating network that learns how much to rely on each model based on user-specific characteristics, such as interaction history length and item popularity.

## Dataset
MovieLens-1M, containing approximately one million user-movie interactions.

## Evaluation
Models are evaluated using Recall@10, Recall@20, NDCG@10, and NDCG@20, along with user-group performance, recommendation diversity, and popularity bias.

## Running

```bash
pip install -r requirements.txt
python -m src.preprocess            # data/raw/ml-1m -> data/processed (chronological leave-two-out)
python -m src.popularity            # most-popular baseline
python -m src.train --model ngcf    # also: sasrec, deepfm (early stopping on validation NDCG@10)
python -m src.compare               # summary, overlap, oracle, per-group tables -> results/
python -m src.plots                 # slide figures -> results/figures/
python -m pytest tests
```

An interrupted training run can continue from its last validation check with `--resume`.
To check how much SASRec relies on the order of same-timestamp ratings:

```bash
python -m src.preprocess --tie-break random --out data/processed_tiebreak_random
python -m src.train --model sasrec --data data/processed_tiebreak_random --out results/tiebreak_random
```

Each `src.train` run writes `results/<model>_seed<seed>.csv` (validation + test metrics),
`_top20.npz` (top-20 lists for both splits), `_log.json` (validation curve) and `.pt` (best weights).
Per-model defaults (epochs, learning rate, batch size, ...) are in `DEFAULTS` in `src/train.py`
and can be overridden on the command line. Add `--save-scores` to keep full score matrices for fusion.

## Team Members
- Mahek Viradiya
- Snigdha Challapalli
- Srushti Thakar

## Project Status
In development — baseline implementation and preliminary experiments.
