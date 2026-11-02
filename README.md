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

## Team Members
- Mahek Viradiya
- Snigdha Challapalli
- Srushti Thakar

## Project Status
In development — baseline implementation and preliminary experiments.
