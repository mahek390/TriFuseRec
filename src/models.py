"""Minimal research baselines (educational implementations, not paper-exact reproductions)."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    """FM + MLP on categorical user ID, user demographics, item ID, movie genres.
    Uses mean-pooled genre embeddings so multi-genre items are supported.
    """
    def __init__(self,n_users,n_items,n_age,n_gender,n_job,n_genres,d=32):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(n,d) for n in [n_users,n_items,n_age,n_gender,n_job,n_genres]])
        self.lin=nn.ModuleList([nn.Embedding(n,1) for n in [n_users,n_items,n_age,n_gender,n_job,n_genres]])
        for e in self.embs: nn.init.normal_(e.weight,std=.01)
        for e in self.lin: nn.init.zeros_(e.weight)
        self.mlp=nn.Sequential(nn.Linear(6*d,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.bias=nn.Parameter(torch.zeros(()))

    def forward(self,u,i,age,gender,job,genre):
        ids=[u,i,age,gender,job]
        e=[emb(x) for emb,x in zip(self.embs[:5],ids)]
        l=[emb(x).squeeze(-1) for emb,x in zip(self.lin[:5],ids)]
        counts=genre.sum(dim=1,keepdim=True).clamp_min(1.)
        g=genre@self.embs[5].weight/counts
        gl=(genre@self.lin[5].weight).squeeze(-1)/counts.squeeze(-1)
        e.append(g);l.append(gl)
        stack=torch.stack(e,dim=1)
        fm=0.5*((stack.sum(1)**2-(stack**2).sum(1)).sum(-1))
        return fm+torch.stack(l,dim=1).sum(1)+self.mlp(torch.cat(e,dim=1)).squeeze(-1)+self.bias


class FastDropout(nn.Module):
    """Same as nn.Dropout, but about 2.5x faster on CPU (Windows), where it dominates SASRec time."""
    def __init__(self,p):
        super().__init__();self.p=p
    def forward(self,x):
        if not self.training or self.p==0:return x
        return x*(torch.rand_like(x)>=self.p)/(1-self.p)


class SASRecBlock(nn.Module):
    """Pre-norm causal self-attention + point-wise feed-forward (as in the SASRec paper).
    Dropout is applied to sublayer outputs, not attention weights (that is very slow on CPU).
    """
    def __init__(self,d,heads,dropout):
        super().__init__()
        self.heads=heads
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d);self.ffn=nn.Sequential(nn.Linear(d,d),nn.ReLU(),FastDropout(dropout),nn.Linear(d,d))
        self.drop=FastDropout(dropout)

    def forward(self,x,allowed):
        b,l,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).view(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed).transpose(1,2).reshape(b,l,d)
        x=x+self.drop(self.out(a))
        return x+self.drop(self.ffn(self.ln2(x)))


class SASRec(nn.Module):
    """Causal Transformer, next-item prediction. Item IDs shifted +1; 0 is left padding.

    The attention mask lets every position attend to itself, so no row is fully masked.
    (The old version combined a causal mask with a key-padding mask: padded positions then
    had every key masked, and PyTorch's eval fast path returned NaN for short histories.)
    """
    def __init__(self,n_items,d=64,max_len=200,heads=2,layers=2,dropout=.2):
        super().__init__()
        self.max_len=max_len;self.d=d
        self.item=nn.Embedding(n_items+1,d,padding_idx=0)
        self.pos=nn.Embedding(max_len,d)
        nn.init.normal_(self.item.weight,std=.02);nn.init.normal_(self.pos.weight,std=.02)
        with torch.no_grad():self.item.weight[0].zero_()
        self.drop=FastDropout(dropout)
        self.blocks=nn.ModuleList([SASRecBlock(d,heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)

    @staticmethod
    def attention_mask(seq):
        """[batch, 1, len, len] boolean, True = may attend: earlier/same real items, plus self."""
        l=seq.shape[1]
        causal=torch.tril(torch.ones(l,l,device=seq.device,dtype=torch.bool))[None]
        allowed=causal&seq.gt(0)[:,None,:]
        allowed=allowed|torch.eye(l,device=seq.device,dtype=torch.bool)[None]
        return allowed[:,None]

    def hidden(self,seq):
        """Hidden state at every position: [batch, len, d]."""
        l=seq.shape[1]
        x=self.item(seq)*math.sqrt(self.d)+self.pos(torch.arange(l,device=seq.device))[None]
        x=self.drop(x)*seq.gt(0).unsqueeze(-1)
        allowed=self.attention_mask(seq)
        for block in self.blocks:x=block(x,allowed)
        return self.norm(x)

    def logits(self,h):
        return h@self.item.weight[1:].T

    def forward(self,seq):
        """Scores for the next item after the (right-aligned) sequence: [batch, n_items]."""
        return self.logits(self.hidden(seq)[:,-1,:])


class SymmetricSpMM(torch.autograd.Function):
    """adj @ x for a symmetric CSR adjacency. The gradient is adj @ grad, which avoids
    PyTorch transposing the sparse matrix on every backward pass (very slow on CPU)."""
    @staticmethod
    def forward(ctx,adj,x):
        ctx.adj=adj
        return adj@x

    @staticmethod
    def backward(ctx,grad):
        return None,ctx.adj@grad


class NGCF(nn.Module):
    """NGCF-style message passing with bi-interaction term and BPR loss.
    Adjacency normalized symmetrically; no external graph dependency.
    """
    def __init__(self,n_users,n_items,edges,d=64,layers=3,reg=1e-5):
        super().__init__()
        self.n_users=n_users;self.n_items=n_items;self.reg=reg
        self.user=nn.Embedding(n_users,d);self.item=nn.Embedding(n_items,d)
        nn.init.xavier_uniform_(self.user.weight);nn.init.xavier_uniform_(self.item.weight)
        self.linear=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        self.bi=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(layers)])
        u=torch.as_tensor(edges[:,0],dtype=torch.long)
        i=torch.as_tensor(edges[:,1],dtype=torch.long)+n_users
        src=torch.cat([u,i]);dst=torch.cat([i,u]);n=n_users+n_items
        degree=torch.bincount(src,minlength=n).float().clamp_min(1)
        values=degree[src].rsqrt()*degree[dst].rsqrt()
        adj=torch.sparse_coo_tensor(torch.stack([src,dst]),values,(n,n)).coalesce()
        # Rebuilt from the training edges on construction, so it is not saved in checkpoints.
        self.register_buffer('adj',adj.to_sparse_csr(),persistent=False)

    def encode(self):
        x=torch.cat([self.user.weight,self.item.weight]);layers=[x]
        for w1,w2 in zip(self.linear,self.bi):
            neighbor=SymmetricSpMM.apply(self.adj,x)
            x=F.leaky_relu(w1(neighbor+x)+w2(x*neighbor),negative_slope=.2)
            layers.append(F.normalize(x,p=2,dim=-1))
        z=torch.cat(layers,dim=-1)
        return z[:self.n_users],z[self.n_users:]

    def bpr(self,u,i,j):
        users,items=self.encode()
        pos=(users[u]*items[i]).sum(-1)
        neg=(users[u]*items[j]).sum(-1)
        reg=self.reg*(self.user(u).square().sum()+self.item(i).square().sum()+self.item(j).square().sum())/len(u)
        return F.softplus(neg-pos).mean()+reg
