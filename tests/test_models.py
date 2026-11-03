import numpy as np
import pandas as pd
import torch
from src.common import NegativeSampler
from src.models import SASRec


def test_sasrec_short_history_no_nan():
    """Regression test: left-padded short histories used to give NaN scores in eval mode."""
    model=SASRec(n_items=20,d=16,max_len=10,heads=2).eval()
    seq=torch.zeros(3,10,dtype=torch.long)
    seq[0,-1:]=torch.tensor([5]);seq[1,-4:]=torch.tensor([1,2,3,4]);seq[2]=torch.arange(1,11)
    with torch.no_grad():
        scores=model(seq)
    assert scores.shape==(3,20)
    assert not torch.isnan(scores).any()


def test_sasrec_ignores_padding_positions():
    """The last position's output must not depend on how much padding precedes it."""
    model=SASRec(n_items=20,d=16,max_len=10,heads=2).eval()
    with torch.no_grad():model.pos.weight.zero_()  # position-free, so only padding could differ
    padded=torch.tensor([[0,0,0,0,0,0,0,3,4,5]]);unpadded=torch.tensor([[3,4,5]])
    shorter=torch.tensor([[0,0,0,0,0,0,0,0,4,5]])
    with torch.no_grad():
        assert torch.allclose(model(padded),model(unpadded),atol=1e-5)
        assert not torch.allclose(model(padded),model(shorter))  # real history does matter


def test_negative_sampler_excludes_seen():
    train=pd.DataFrame({'user_idx':[0,0,0,1],'item_idx':[0,1,2,3]})
    sampler=NegativeSampler(train,n_items=5)
    items=sampler.sample(np.zeros(1000,dtype=np.int64),np.random.default_rng(0))
    assert set(items.tolist())<={3,4}
    items=sampler.sample(np.ones(1000,dtype=np.int64),np.random.default_rng(0))
    assert 3 not in set(items.tolist())
