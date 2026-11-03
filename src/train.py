"""Train one baseline with early stopping on validation, then evaluate validation and test.

python -m src.train --model ngcf
python -m src.train --model sasrec
python -m src.train --model deepfm

Model selection uses validation NDCG@10 only; test numbers are never used for choices.
For test ranking, the validation item is added to the user's history (SASRec input) and
masked as seen; the graph and DeepFM still use training interactions only.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .common import seed_everything, read_data, rank_metrics, NegativeSampler
from .models import DeepFM,SASRec,NGCF


DEFAULTS={  # per-model defaults; any of these can be overridden on the command line
    'deepfm':dict(dim=32,lr=1e-3,batch=2048,epochs=40,eval_every=2,patience=4,neg=4,weight_decay=0.),
    'sasrec':dict(dim=64,lr=1e-3,batch=128,epochs=200,eval_every=10,patience=3,neg=0,weight_decay=0.),
    'ngcf':dict(dim=64,lr=5e-3,batch=8192,epochs=150,eval_every=5,patience=4,neg=1,weight_decay=0.),
}


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


def left_pad(items,max_len):
    """Shift IDs by +1 (0 = padding) and right-align the last max_len items."""
    seq=np.zeros(max_len,dtype=np.int64)
    tail=items[-max_len:]
    if len(tail):seq[-len(tail):]=np.asarray(tail)+1
    return seq


def sasrec_batches(histories,max_len,batch,rng):
    """One random window of up to max_len+1 items per user; predict every next item."""
    users=rng.permutation(len(histories))
    for start in range(0,len(users),batch):
        inputs=[];targets=[]
        for u in users[start:start+batch]:
            h=histories[u]
            if len(h)>max_len+1:
                s=int(rng.integers(0,len(h)-max_len))
                h=h[s:s+max_len+1]
            inputs.append(left_pad(h[:-1],max_len))
            t=np.full(max_len,-100,dtype=np.int64);t[-(len(h)-1):]=h[1:]
            targets.append(t)
        yield torch.from_numpy(np.stack(inputs)),torch.from_numpy(np.stack(targets))


class Runner:
    """Holds a model plus its per-epoch training step and full-catalog scoring function."""
    def __init__(self,args,path,device,tr,histories,nu,ni):
        self.args=args;self.device=device;self.histories=histories;self.nu=nu;self.ni=ni
        self.rng=np.random.default_rng(args.seed)
        self.positives=tr[['user_idx','item_idx']].to_numpy(dtype=np.int64)
        self.sampler=NegativeSampler(tr,ni)
        if args.model=='deepfm':
            self.uf,self.gf,cats=features(path,nu,ni,device)
            self.model=DeepFM(nu,ni,*cats,d=args.dim).to(device)
        elif args.model=='sasrec':
            self.model=SASRec(ni,d=args.dim,max_len=args.max_len,dropout=args.dropout).to(device)
        else:
            self.model=NGCF(nu,ni,self.positives,d=args.dim,layers=args.layers).to(device)
        self.opt=torch.optim.Adam(self.model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    def deepfm_logit(self,u,i):
        f=self.uf[u]
        return self.model(u,i,f[:,0],f[:,1],f[:,2],self.gf[i])

    def train_epoch(self):
        a=self.args;self.model.train();losses=[]
        if a.model=='sasrec':
            for seq,tgt in sasrec_batches(self.histories,a.max_len,a.batch,self.rng):
                seq=seq.to(self.device);tgt=tgt.to(self.device)
                keep=tgt.ne(-100)
                h=self.model.hidden(seq)[keep]
                loss=F.cross_entropy(self.model.logits(h),tgt[keep])
                self.opt.zero_grad();loss.backward();self.opt.step();losses.append(loss.item())
            return float(np.mean(losses))
        order=self.rng.permutation(len(self.positives))
        for start in range(0,len(order),a.batch):
            pu,pi=self.positives[order[start:start+a.batch]].T
            if a.model=='ngcf':
                nj=self.sampler.sample(pu,self.rng)
                u,i,j=[torch.as_tensor(x,device=self.device) for x in (pu,pi,nj)]
                loss=self.model.bpr(u,i,j)
            else:  # DeepFM: pointwise BCE, a.neg fresh negatives per positive
                nu_=np.repeat(pu,a.neg);nj=self.sampler.sample(nu_,self.rng)
                u=torch.as_tensor(np.concatenate([pu,nu_]),device=self.device)
                i=torch.as_tensor(np.concatenate([pi,nj]),device=self.device)
                y=torch.cat([torch.ones(len(pu)),torch.zeros(len(nu_))]).to(self.device)
                loss=F.binary_cross_entropy_with_logits(self.deepfm_logit(u,i),y)
            self.opt.zero_grad();loss.backward();self.opt.step();losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def scores(self,histories):
        """Full [users, items] score matrix given each user's input history."""
        a=self.args;self.model.eval()
        out=np.empty((self.nu,self.ni),dtype=np.float32)
        if a.model=='ngcf':
            eu,ei=self.model.encode()
            for s in range(0,self.nu,512):out[s:s+512]=(eu[s:s+512]@ei.T).cpu().numpy()
        elif a.model=='sasrec':
            for s in range(0,self.nu,256):
                seq=np.stack([left_pad(histories[u],a.max_len) for u in range(s,min(s+256,self.nu))])
                out[s:s+256]=self.model(torch.from_numpy(seq).to(self.device)).cpu().numpy()
        else:
            items=torch.arange(self.ni,device=self.device)
            for s in range(0,self.nu,32):
                users=torch.arange(s,min(s+32,self.nu),device=self.device)
                u=users.repeat_interleave(self.ni);i=items.repeat(len(users))
                out[s:s+len(users)]=self.deepfm_logit(u,i).view(len(users),self.ni).cpu().numpy()
        if np.isnan(out).any():raise RuntimeError('NaN scores')
        return out


def train(args):
    for k,v in DEFAULTS[args.model].items():
        if getattr(args,k) is None:setattr(args,k,v)
    seed_everything(args.seed)
    if args.threads:torch.set_num_threads(args.threads)
    path=Path(args.data);output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr,histories,valid,test,nu,ni=read_data(path)
    test_histories=[h+[int(v)] for h,v in zip(histories,valid)]
    run=Runner(args,path,device,tr,histories,nu,ni)
    print(f'{args.model} config:',{k:getattr(args,k) for k in sorted(DEFAULTS[args.model])},flush=True)

    tag=f'{args.model}_seed{args.seed}'
    ckpt_path=output/f'{tag}_ckpt.pt'
    best=(-1.,0,None);log=[];bad=0;start=1;t0=time.time();stopped=False
    if args.resume and ckpt_path.exists():
        ck=torch.load(ckpt_path,weights_only=False)
        run.model.load_state_dict(ck['model']);run.opt.load_state_dict(ck['opt'])
        run.rng.bit_generator.state=ck['np_rng'];torch.set_rng_state(ck['torch_rng'])
        best,log,bad,start,stopped=ck['best'],ck['log'],ck['bad'],ck['epoch']+1,ck['stopped']
        t0-=log[-1]['minutes']*60 if log else 0
        print(f'resumed from {ckpt_path} after epoch {ck["epoch"]} (best epoch {best[1]})',flush=True)
    for epoch in range(start,args.epochs+1):
        if stopped:break
        loss=run.train_epoch()
        if epoch%args.eval_every and epoch!=args.epochs:continue
        m,_=rank_metrics(run.scores(histories),valid,histories)
        log.append({'epoch':epoch,'loss':loss,**m,'minutes':(time.time()-t0)/60})
        print(f"{args.model} epoch {epoch} loss={loss:.4f} val R@10={m['Recall@10']:.4f} "
              f"N@10={m['NDCG@10']:.4f} ({(time.time()-t0)/60:.1f} min)",flush=True)
        if m['NDCG@10']>best[0]:best=(m['NDCG@10'],epoch,copy.deepcopy(run.model.state_dict()));bad=0
        else:
            bad+=1
            if bad>=args.patience:print('early stop',flush=True);stopped=True
        # Checkpoint at every validation check so an interrupted run can --resume.
        tmp=ckpt_path.with_suffix('.tmp')
        torch.save({'model':run.model.state_dict(),'opt':run.opt.state_dict(),'epoch':epoch,'best':best,
                    'log':log,'bad':bad,'stopped':stopped,'np_rng':run.rng.bit_generator.state,
                    'torch_rng':torch.get_rng_state()},tmp)
        tmp.replace(ckpt_path)
    run.model.load_state_dict(best[2])

    rows=[];npz={}
    for split,hists,truth in [('validation',histories,valid),('test',test_histories,test)]:
        s=run.scores(hists)
        metrics,top=rank_metrics(s,truth,hists)
        rows.append({'model':args.model,'seed':args.seed,'split':split,'best_epoch':best[1],**metrics})
        npz[f'top20_{split}']=top;npz[f'truth_{split}']=truth
        if args.save_scores:
            np.save(output/f'{args.model}_seed{args.seed}_scores_{split}.npy',s.astype(np.float16))
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(output/f'{tag}.csv',index=False)
    np.savez_compressed(output/f'{tag}_top20.npz',**npz)
    (output/f'{tag}_log.json').write_text(json.dumps({'config':vars(args),'evals':log},indent=1,default=str))
    torch.save(run.model.state_dict(),output/f'{tag}.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['deepfm','sasrec','ngcf'])
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='results')
    p.add_argument('--seed',type=int,default=42)
    for name,typ in [('epochs',int),('dim',int),('batch',int),('eval-every',int),('patience',int),
                     ('neg',int),('lr',float),('weight-decay',float)]:
        p.add_argument(f'--{name}',type=typ,default=None,help='default depends on --model')
    p.add_argument('--max-len',type=int,default=200)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--layers',type=int,default=3,help='NGCF propagation layers')
    p.add_argument('--threads',type=int,default=0)
    p.add_argument('--save-scores',action='store_true',help='save full float16 score matrices (for fusion)')
    p.add_argument('--resume',action='store_true',help='continue from results/<model>_seed<seed>_ckpt.pt')
    p.add_argument('--cpu',action='store_true')
    train(p.parse_args())
