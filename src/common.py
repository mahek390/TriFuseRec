"""Shared data, reproducibility, ranking metrics, and negative sampling."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_data(path):
    path=Path(path)
    train=pd.read_csv(path/'train.csv')
    valid=pd.read_csv(path/'valid.csv')
    test=pd.read_csv(path/'test.csv')
    n_users=int(pd.read_csv(path/'user_map.csv').shape[0])
    n_items=int(pd.read_csv(path/'item_map.csv').shape[0])
    histories=[[] for _ in range(n_users)]
    # Stable sort keeps the file's order for equal timestamps (set by preprocess --tie-break).
    for r in train.sort_values(['user_idx','timestamp'],kind='stable').itertuples():
        histories[int(r.user_idx)].append(int(r.item_idx))
    val_items=valid.sort_values('user_idx').item_idx.to_numpy(dtype=np.int64)
    test_items=test.sort_values('user_idx').item_idx.to_numpy(dtype=np.int64)
    assert len(val_items)==len(test_items)==n_users
    return train,histories,val_items,test_items,n_users,n_items


class NegativeSampler:
    """Uniformly samples movies a user has not interacted with in training (vectorized)."""
    def __init__(self, train, n_items):
        self.n_items=n_items
        self.keys=np.unique(train.user_idx.to_numpy(np.int64)*n_items+train.item_idx.to_numpy(np.int64))

    def _seen(self, users, items):
        k=users*self.n_items+items
        pos=np.minimum(np.searchsorted(self.keys,k),len(self.keys)-1)
        return self.keys[pos]==k

    def sample(self, users, rng):
        users=np.asarray(users,dtype=np.int64)
        items=rng.integers(self.n_items,size=len(users))
        bad=self._seen(users,items)
        while bad.any():
            items[bad]=rng.integers(self.n_items,size=int(bad.sum()))
            bad[bad]=self._seen(users[bad],items[bad])
        return items


def rank_metrics(scores, truths, seen, ks=(10,20)):
    """scores [users,items], truth [users], seen list of sets/lists to exclude.

    Every held-out target must remain an eligible candidate.
    """
    scores=np.array(scores,copy=True)
    n_users,n_items=scores.shape
    assert len(truths)==len(seen)==n_users
    for u,items in enumerate(seen):
        if int(truths[u]) in items:
            raise ValueError(f'Target movie is in masked history for user {u}')
        scores[u,list(items)]=-np.inf
    max_k=min(max(ks),n_items)
    # Full ranking is feasible for ML-1M and has deterministic tie handling.
    order=np.argsort(-scores,axis=1,kind='stable')[:,:max_k]
    metrics={}
    for k in ks:
        matches=(order[:,:min(k,n_items)]==np.asarray(truths)[:,None])
        hits=matches.any(axis=1)
        ranks=np.argmax(matches,axis=1)+1
        ndcg=np.where(hits,1.0/np.log2(ranks+1),0.)
        metrics[f'Recall@{k}']=float(np.mean(hits))
        metrics[f'NDCG@{k}']=float(np.mean(ndcg))
    return metrics,order


def evaluate_model(score_fn, histories, truths, extra_seen=None, batch_size=128):
    all_scores=[]
    n=len(histories)
    for start in range(0,n,batch_size):
        users=np.arange(start,min(start+batch_size,n))
        all_scores.append(score_fn(users))
    scores=np.concatenate(all_scores)
    seen=[set(h) for h in histories]
    if extra_seen is not None:
        for u,it in enumerate(extra_seen): seen[u].add(int(it))
    metrics,order=rank_metrics(scores,truths,seen)
    return metrics,order
