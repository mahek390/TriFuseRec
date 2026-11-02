import numpy as np
from src.common import rank_metrics

def test_rank_metrics():
    scores=np.array([[.1,.9,.8,.2],[.8,.1,.7,.6]])
    metrics,order=rank_metrics(scores,np.array([2,1]),[{0},{0}],[1,2,3])
    assert metrics['Recall@1']==0
    assert metrics['Recall@2']==.5
    assert metrics['Recall@3']==1.
    assert np.isclose(metrics['NDCG@2'],.5/np.log2(3))
