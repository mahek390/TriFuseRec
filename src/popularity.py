"""Sanity-check baseline: rank movies by training popularity."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_data, rank_metrics


def run(data, out):
    train,histories,valid,test,n_users,n_items=read_data(data)
    popularity=np.bincount(train.item_idx.to_numpy(),minlength=n_items).astype(float)
    scores=np.broadcast_to(popularity,(n_users,n_items)).copy()
    test_histories=[h+[int(v)] for h,v in zip(histories,valid)]
    rows=[];npz={}
    for split,hists,truth in [('validation',histories,valid),('test',test_histories,test)]:
        metrics,top=rank_metrics(scores,truth,hists)
        rows.append({'model':'popularity','seed':0,'split':split,**metrics})
        npz[f'top20_{split}']=top;npz[f'truth_{split}']=truth
    Path(out).mkdir(parents=True,exist_ok=True)
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(Path(out)/'popularity.csv',index=False)
    np.savez_compressed(Path(out)/'popularity_top20.npz',**npz)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='results')
    a=p.parse_args();run(a.data,a.out)
