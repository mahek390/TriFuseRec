"""Week 1 sanity-check baseline (training popularity only)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_data, rank_metrics


def run(data, out):
    train,histories,valid,test,n_users,n_items=read_data(data)
    popularity=np.bincount(train.item_idx.to_numpy(),minlength=n_items).astype(float)
    scores=np.broadcast_to(popularity,(n_users,n_items)).copy()
    metrics,top=rank_metrics(scores,valid,histories)
    print('VALIDATION',metrics)
    Path(out).mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'model':'MostPopular','split':'validation',**metrics}]).to_csv(Path(out)/'popularity.csv',index=False)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='results')
    a=p.parse_args();run(a.data,a.out)
