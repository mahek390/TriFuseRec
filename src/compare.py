"""Preliminary analysis of the baselines (run after src.popularity and src.train).

python -m src.compare             # test split (models were selected on validation)
python -m src.compare --split validation

Writes CSVs and results/preliminary_report_<split>.md containing:
  summary table, oracle (best model per user), overlap of users each model hits,
  Recall/NDCG@10 by user history length and by target-item popularity,
  popularity bias (share of top-10 from the 20% most popular movies) and catalog coverage.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_data

MODELS=('deepfm','sasrec','ngcf')
K=10


def hit_and_ndcg(top,truth,k=K):
    match=top[:,:k]==truth[:,None]
    hit=match.any(1)
    ndcg=np.where(hit,1/np.log2(np.argmax(match,1)+2),0.)
    return hit,ndcg


def md(df,floatfmt='.4f'):
    """Small markdown table writer (avoids the optional tabulate dependency)."""
    cols=list(df.columns)
    fmt=lambda v:format(v,floatfmt) if isinstance(v,(float,np.floating)) else str(v)
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join('---' for _ in cols)+'|']
    lines+=['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def main(data,out,seed,split):
    out=Path(out)
    train,histories,valid,test,n_users,n_items=read_data(data)
    truth=valid if split=='validation' else test
    hist_len=np.array([len(h) for h in histories])
    popularity=np.bincount(train.item_idx.to_numpy(),minlength=n_items)
    head=np.zeros(n_items,dtype=bool);head[np.argsort(-popularity,kind='stable')[:int(.2*n_items)]]=True

    files={'popularity':out/'popularity_top20.npz',**{m:out/f'{m}_seed{seed}_top20.npz' for m in MODELS}}
    tops={}
    for name,p in files.items():
        if not p.exists():print(f'missing {p}, skipping {name}');continue
        z=np.load(p)
        if f'top20_{split}' not in z:print(f'{p} has old format, retrain {name}');continue
        assert np.array_equal(z[f'truth_{split}'],truth)
        tops[name]=z[f'top20_{split}']
    hits={};ndcgs={}
    for name,top in tops.items():hits[name],ndcgs[name]=hit_and_ndcg(top,truth)
    neural=[m for m in MODELS if m in tops]
    report=[f'# Preliminary results ({split} split, seed {seed})\n',
            f'MovieLens-1M, {n_users} users, {n_items} movies, full-catalog ranking. '
            f'One held-out movie per user, so Recall@K equals HitRate@K.\n']

    # 1. Summary table
    csvs=[out/'popularity.csv']+[out/f'{m}_seed{seed}.csv' for m in MODELS]
    summary=pd.concat([pd.read_csv(p) for p in csvs if p.exists()],ignore_index=True)
    summary.to_csv(out/'baseline_summary.csv',index=False)
    cols=['model','Recall@10','NDCG@10','Recall@20','NDCG@20']+(['best_epoch'] if 'best_epoch' in summary else [])
    shown=summary[summary.split==split][cols].copy()
    if 'best_epoch' in shown:shown['best_epoch']=[('-' if pd.isna(v) else str(int(v))) for v in shown.best_epoch]
    report+=['## Main results\n',md(shown),'']

    # 2. Oracle and overlap
    if len(neural)>1:
        H=np.stack([hits[m] for m in neural],1)
        best_single=max(neural,key=lambda m:ndcgs[m].mean())
        oracle=pd.DataFrame([
            {'what':f'best single model ({best_single})','Recall@10':hits[best_single].mean(),'NDCG@10':ndcgs[best_single].mean()},
            {'what':'oracle: best of '+'/'.join(neural)+' per user','Recall@10':H.any(1).mean(),
             'NDCG@10':np.max([ndcgs[m] for m in neural],0).mean()}])
        rows=[]
        for bits in range(1,2**len(neural)):
            chosen=np.array([(bits>>i)&1 for i in range(len(neural))],dtype=bool)
            rows.append({'models that hit (top-10)':' + '.join(np.array(neural)[chosen]),
                         'users':int((H==chosen).all(1).sum())})
        rows.append({'models that hit (top-10)':'none','users':int((~H).all(1).sum())})
        overlap=pd.DataFrame(rows)
        overlap.to_csv(out/f'overlap_{split}.csv',index=False)
        oracle.to_csv(out/f'oracle_{split}.csv',index=False)
        report+=['## Headroom for per-user fusion\n',md(oracle),'',
                 '## Which models get each user right (top-10)\n',md(overlap),'']

    # 3. By user history length (training interactions, quartiles)
    q=np.quantile(hist_len,[.25,.5,.75]).astype(int)
    groups=np.digitize(hist_len,q,right=True)
    edges=[hist_len.min(),*q,hist_len.max()]
    labels=[f'{edges[0]}-{edges[1]}']+[f'{edges[i]+1}-{edges[i+1]}' for i in range(1,4)]
    rows=[]
    for g in range(4):
        s=groups==g
        row={'history length':labels[g],'users':int(s.sum())}
        for name in tops:row[f'{name} R@10']=hits[name][s].mean()
        for name in tops:row[f'{name} N@10']=ndcgs[name][s].mean()
        if len(neural)>1:row['oracle R@10']=np.stack([hits[m][s] for m in neural]).any(0).mean()
        rows.append(row)
    by_hist=pd.DataFrame(rows);by_hist.to_csv(out/f'by_history_{split}.csv',index=False)
    show=['history length','users']+[c for c in by_hist if c.endswith('R@10')]
    report+=['## Recall@10 by user history length (quartiles of training interactions)\n',md(by_hist[show]),'']

    # 4. By target-item popularity
    target_head=head[truth]
    rows=[]
    for label,s in [('head (top 20% popular)',target_head),('tail (other 80%)',~target_head)]:
        row={'held-out movie':label,'users':int(s.sum())}
        for name in tops:row[f'{name} R@10']=hits[name][s].mean()
        rows.append(row)
    by_pop=pd.DataFrame(rows);by_pop.to_csv(out/f'by_item_popularity_{split}.csv',index=False)
    report+=['## Recall@10 by popularity of the held-out movie\n',md(by_pop),'']

    # 5. Popularity bias and coverage
    rows=[]
    for name,top in tops.items():
        t=top[:,:K]
        rows.append({'model':name,'head share of top-10':head[t].mean(),
                     'catalog coverage@10':len(np.unique(t))/n_items})
    bias=pd.DataFrame(rows);bias.to_csv(out/f'popularity_bias_{split}.csv',index=False)
    report+=['## Popularity bias and coverage\n',
             f'For reference, {target_head.mean():.1%} of held-out movies are in the head.\n',md(bias,'.3f'),'']

    text='\n'.join(report)
    (out/f'preliminary_report_{split}.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed')
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--split',default='test',choices=['validation','test'])
    a=ap.parse_args();main(a.data,a.out,a.seed,a.split)
