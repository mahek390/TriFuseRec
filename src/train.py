"""Run a single Week 1 baseline and evaluate validation full-catalog rankings.

python -m src.train --model sasrec --epochs 3
python -m src.train --model deepfm --epochs 3
python -m src.train --model ngcf --epochs 3
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .common import seed_everything, read_data, rank_metrics, negative_samples
from .models import DeepFM,SASRec,NGCF


def features(path,n_users,n_items,device):
    users=pd.read_csv(path/'users.csv')
    movies=pd.read_csv(path/'movies.csv')
    ages=sorted(users.age.unique());genders=sorted(users.gender.unique());jobs=sorted(users.occupation.unique())
    a={v:k for k,v in enumerate(ages)};g={v:k for k,v in enumerate(genders)};o={v:k for k,v in enumerate(jobs)}
    user_f=np.zeros((n_users,3),dtype=np.int64)
    for r in users.itertuples():user_f[int(r.user_idx)]=[a[r.age],g[r.gender],o[r.occupation]]
    genre_names=sorted({genre for genres in movies.genres for genre in genres.split('|')})
    genre_map={v:k for k,v in enumerate(genre_names)}
    genre_f=np.zeros((n_items,len(genre_names)),dtype=np.float32)
    for r in movies.itertuples():
        for genre in r.genres.split('|'):genre_f[int(r.item_idx),genre_map[genre]]=1
    return (torch.from_numpy(user_f).to(device),torch.from_numpy(genre_f).to(device),
            [len(ages),len(genders),len(jobs),len(genre_names)])


def sequence_rows(histories,max_len):
    inputs=[];targets=[]
    for h in histories:
        for t in range(1,len(h)):
            inputs.append(([0]*max(0,max_len-t)+[x+1 for x in h[max(0,t-max_len):t]])[-max_len:])
            targets.append(h[t])
    return torch.tensor(inputs,dtype=torch.long),torch.tensor(targets,dtype=torch.long)


def train(args):
    seed_everything(args.seed)
    path=Path(args.data);output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr,histories,valid,test,nu,ni=read_data(path)
    if args.model=='deepfm':
        uf,gf,cats=features(path,nu,ni,device)
        model=DeepFM(nu,ni,*cats,d=args.dim).to(device)
        us,its,ys=negative_samples(tr,ni,args.neg,args.seed)
        loader=DataLoader(TensorDataset(torch.from_numpy(us),torch.from_numpy(its),torch.from_numpy(ys)),batch_size=args.batch,shuffle=True)
    elif args.model=='sasrec':
        model=SASRec(ni,d=args.dim,max_len=args.max_len).to(device)
        sequences,labels=sequence_rows(histories,args.max_len)
        loader=DataLoader(TensorDataset(sequences,labels),batch_size=args.batch,shuffle=True)
    else:
        edges=tr[['user_idx','item_idx']].to_numpy(dtype=np.int64)
        model=NGCF(nu,ni,edges,d=args.dim).to(device)
        # NGCF propagates across the entire graph for each optimizer step.
        loader=None
        positives=tr[['user_idx','item_idx']].to_numpy(dtype=np.int64)
        seen=[set(h) for h in histories]
        rng=np.random.default_rng(args.seed)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for epoch in range(args.epochs):
        model.train();losses=[]
        if args.model=='ngcf':
            rng.shuffle(positives)
            for chunk in np.array_split(positives,max(1,int(np.ceil(len(positives)/args.graph_batch)))):
                u=chunk[:,0];i=chunk[:,1]
                j=rng.integers(ni,size=len(u))
                for idx in range(len(u)):
                    while int(j[idx]) in seen[int(u[idx])]: j[idx]=rng.integers(ni)
                ut=torch.as_tensor(u,device=device);it=torch.as_tensor(i,device=device);jt=torch.as_tensor(j,device=device)
                opt.zero_grad();loss=model.bpr(ut,it,jt);loss.backward();opt.step();losses.append(float(loss.item()))
        else:
            for batch in loader:
                opt.zero_grad()
                if args.model=='deepfm':
                    u,i,y=[x.to(device) for x in batch]
                    f=uf[u];logit=model(u,i,f[:,0],f[:,1],f[:,2],gf[i])
                    loss=nn.functional.binary_cross_entropy_with_logits(logit,y)
                else:
                    seq,y=[x.to(device) for x in batch]
                    loss=nn.functional.cross_entropy(model(seq),y)
                loss.backward();opt.step();losses.append(float(loss.item()))
        print(f'{args.model} epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f}',flush=True)
    model.eval()
    scores=np.empty((nu,ni),dtype=np.float32)
    with torch.no_grad():
        if args.model=='ngcf':
            eu,ei=model.encode()
            for start in range(0,nu,args.eval_batch):
                scores[start:start+args.eval_batch]=(eu[start:start+args.eval_batch]@ei.T).cpu().numpy()
        else:
            for start in range(0,nu,args.eval_batch):
                end=min(start+args.eval_batch,nu)
                if args.model=='sasrec':
                    seq=np.zeros((end-start,args.max_len),dtype=np.int64)
                    for row,u in enumerate(range(start,end)):
                        hist=[i+1 for i in histories[u][-args.max_len:]]
                        seq[row,-len(hist):]=hist
                    scores[start:end]=model(torch.from_numpy(seq).to(device)).cpu().numpy()
                else:
                    for u in range(start,end):
                        chunks=[]
                        for s in range(0,ni,512):
                            ids=torch.arange(s,min(s+512,ni),device=device)
                            uu=torch.full_like(ids,u);f=uf[uu]
                            chunks.append(model(uu,ids,f[:,0],f[:,1],f[:,2],gf[ids]).cpu().numpy())
                        scores[u]=np.concatenate(chunks)
    metrics,top=rank_metrics(scores,valid,histories)
    result={'model':args.model,'seed':args.seed,'split':'validation',**metrics}
    print('VALIDATION:',json.dumps(result,indent=2))
    pd.DataFrame([result]).to_csv(output/f'{args.model}_seed{args.seed}.csv',index=False)
    np.savez_compressed(output/f'{args.model}_seed{args.seed}_top20.npz',top20=top,validation_truth=valid)
    torch.save(model.state_dict(),output/f'{args.model}_seed{args.seed}.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['deepfm','sasrec','ngcf'])
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='results')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dim',type=int,default=32)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--graph-batch',type=int,default=8192)
    p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--max-len',type=int,default=50)
    p.add_argument('--neg',type=int,default=2)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--cpu',action='store_true')
    train(p.parse_args())
