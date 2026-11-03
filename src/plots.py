"""Slide-ready figures from the results (run after src.compare).

python -m src.plots      # writes results/figures/*.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fixed colour per model (matches the proposal diagram); popularity is the neutral baseline.
COLORS={'popularity':'#8a8985','deepfm':'#2a78d6','sasrec':'#1baf7a','ngcf':'#eb6834'}
NAMES={'popularity':'Popularity','deepfm':'DeepFM','sasrec':'SASRec','ngcf':'NGCF'}
ORDER=['popularity','deepfm','sasrec','ngcf']
INK='#0b0b0b';INK2='#52514e';GRID='#e4e3df'

plt.rcParams.update({'font.size':11,'axes.edgecolor':INK2,'axes.labelcolor':INK2,'xtick.color':INK2,
                     'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'axes.grid.axis':'y','grid.color':GRID,'grid.linewidth':.8,
                     'axes.axisbelow':True,'figure.dpi':150,'savefig.bbox':'tight','legend.frameon':False})


def save(fig,out,name):
    fig.savefig(out/name);plt.close(fig);print('wrote',out/name)


def bar_labels(ax,bars,fmt='{:.3f}'):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),(b.get_x()+b.get_width()/2,b.get_height()),
                    xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=9,color=INK)


def main(out,split):
    res=Path(out);fig_dir=res/'figures';fig_dir.mkdir(exist_ok=True)
    summary=pd.read_csv(res/'baseline_summary.csv');summary=summary[summary.split==split]
    models=[m for m in ORDER if m in set(summary.model)]

    # 1. Main results: Recall@10 and NDCG@10 side by side (separate axes, one metric each)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,metric in zip(axes,['Recall@10','NDCG@10']):
        vals=[float(summary[summary.model==m][metric].iloc[0]) for m in models]
        bars=ax.bar([NAMES[m] for m in models],vals,color=[COLORS[m] for m in models],width=.6,edgecolor='white',linewidth=2)
        bar_labels(ax,bars);ax.set_title(metric,color=INK,loc='left');ax.set_ylim(0,max(vals)*1.18)
    fig.suptitle(f'Baselines on MovieLens-1M ({split}, full-catalog ranking)',x=.06,ha='left',color=INK)
    save(fig,fig_dir,'main_results.png')

    # 2. Recall@10 by history-length quartile (grouped bars)
    p=res/f'by_history_{split}.csv'
    if p.exists():
        h=pd.read_csv(p);ms=[m for m in models if f'{m} R@10' in h]
        x=np.arange(len(h));w=.8/len(ms)
        fig,ax=plt.subplots(figsize=(9,3.8))
        for k,m in enumerate(ms):
            ax.bar(x+(k-(len(ms)-1)/2)*w,h[f'{m} R@10'],w,color=COLORS[m],label=NAMES[m],edgecolor='white',linewidth=1.5)
        ax.set_xticks(x,[f'{g}\n({n} users)' for g,n in zip(h['history length'],h.users)])
        ax.set_xlabel('Training interactions per user (quartiles)');ax.set_ylabel('Recall@10')
        ax.set_title('Which view helps which users?',color=INK,loc='left');ax.legend(ncol=len(ms),loc='upper right')
        save(fig,fig_dir,'by_history_length.png')

    # 3. Overlap: users hit by each combination of models (excluding "none")
    p=res/f'overlap_{split}.csv'
    if p.exists():
        o=pd.read_csv(p);none=int(o[o.iloc[:,0]=='none'].users.sum())
        o=o[o.iloc[:,0]!='none'].sort_values('users')
        labels=[' + '.join(NAMES[s.strip()] for s in c.split('+')) for c in o.iloc[:,0]]
        fig,ax=plt.subplots(figsize=(7,.45*len(o)+1.4))
        ax.grid(axis='y',visible=False);ax.grid(axis='x',color=GRID)
        bars=ax.barh(labels,o.users,color='#2a78d6',height=.6)
        for b in bars:ax.annotate(f'{int(b.get_width())}',(b.get_width(),b.get_y()+b.get_height()/2),
                                  xytext=(4,0),textcoords='offset points',va='center',fontsize=9,color=INK)
        ax.set_xlabel('Users whose held-out movie is in the top 10')
        ax.set_title(f'Which models get each user right ({none} users: none)',color=INK,loc='left')
        save(fig,fig_dir,'overlap.png')

    # 4. Oracle headroom
    p=res/f'oracle_{split}.csv'
    if p.exists():
        o=pd.read_csv(p)
        fig,ax=plt.subplots(figsize=(5.5,3.4))
        bars=ax.bar(['Best single\nmodel','Oracle: best\nmodel per user'],o['Recall@10'],color=['#8a8985','#6250d6'],width=.55)
        bar_labels(ax,bars);ax.set_ylabel('Recall@10');ax.set_ylim(0,o['Recall@10'].max()*1.2)
        ax.set_title('Headroom for per-user fusion',color=INK,loc='left')
        save(fig,fig_dir,'oracle_headroom.png')

    # 5. Popularity bias and coverage
    p=res/f'popularity_bias_{split}.csv'
    if p.exists():
        b=pd.read_csv(p).set_index('model');ms=[m for m in ORDER if m in b.index]
        fig,axes=plt.subplots(1,2,figsize=(9,3.4))
        for ax,col,title in zip(axes,['head share of top-10','catalog coverage@10'],
                                ['Share of top-10 from the 20% most popular movies','Catalog coverage of top-10 lists']):
            bars=ax.bar([NAMES[m] for m in ms],b.loc[ms,col],color=[COLORS[m] for m in ms],width=.6)
            bar_labels(ax,bars,'{:.0%}');ax.set_ylim(0,1.15);ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            ax.set_title(title,color=INK,loc='left',fontsize=10)
        save(fig,fig_dir,'popularity_bias.png')

    # 6. Validation curves (one panel per model; scales differ a lot)
    logs={m:json.loads((res/f'{m}_seed42_log.json').read_text())['evals'] for m in ['deepfm','sasrec','ngcf']
          if (res/f'{m}_seed42_log.json').exists()}
    if logs:
        fig,axes=plt.subplots(1,len(logs),figsize=(3.6*len(logs),3.4),squeeze=False,layout='constrained')
        for ax,(m,ev) in zip(axes[0],logs.items()):
            e=[r['epoch'] for r in ev];r=[r['Recall@10'] for r in ev]
            ax.plot(e,r,color=COLORS[m],lw=2,marker='o',ms=4)
            best=int(np.argmax([x['NDCG@10'] for x in ev]))
            ax.plot(e[best],r[best],'o',ms=9,mfc='white',mec=COLORS[m],mew=2)
            ax.set_title(NAMES[m],color=INK,loc='left');ax.set_xlabel('epoch')
        axes[0][0].set_ylabel('Validation Recall@10')
        fig.suptitle('Training curves (ring = checkpoint kept, best validation NDCG@10)',x=.02,ha='left',color=INK,fontsize=11)
        save(fig,fig_dir,'training_curves.png')

    # 7. SASRec tie-break check
    p=res/'tiebreak_random'/'sasrec_seed42_log.json'
    if p.exists() and 'sasrec' in logs:
        fig,ax=plt.subplots(figsize=(6.5,3.6))
        for ev,label,color in [(logs['sasrec'],'ties ordered by movie ID','#1baf7a'),
                               (json.loads(p.read_text())['evals'],'ties shuffled randomly','#6250d6')]:
            ax.plot([r['epoch'] for r in ev],[r['Recall@10'] for r in ev],lw=2,marker='o',ms=4,color=color,label=label)
        ax.set_xlabel('epoch');ax.set_ylabel('Validation Recall@10');ax.set_ylim(0,None);ax.legend(loc='lower right')
        ax.set_title('SASRec: is the sequence signal real?',color=INK,loc='left')
        save(fig,fig_dir,'sasrec_tiebreak.png')


def eda(data,out):
    """Dataset figure: ratings per user and the item-popularity long tail."""
    d=Path(data);fig_dir=Path(out)/'figures';fig_dir.mkdir(parents=True,exist_ok=True)
    al=pd.concat([pd.read_csv(d/f) for f in ['train.csv','valid.csv','test.csv']])
    per_user=al.groupby('user_idx').size();per_item=al.groupby('item_idx').size().sort_values(ascending=False).values
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    ax=axes[0];bins=np.logspace(np.log10(per_user.min()),np.log10(per_user.max()),30)
    ax.hist(per_user,bins=bins,color=COLORS['deepfm'],edgecolor='white',linewidth=1);ax.set_xscale('log')
    ax.axvline(per_user.median(),color=INK,lw=1.5,ls='--')
    ax.annotate(f'median {per_user.median():.0f}',(per_user.median(),ax.get_ylim()[1]*.9),xytext=(6,0),
                textcoords='offset points',fontsize=9,color=INK,bbox=dict(fc='white',ec='none',pad=1))
    ax.set_xlabel('ratings per user (log scale)');ax.set_ylabel('users');ax.set_title('Ratings per user',color=INK,loc='left')
    ax=axes[1];rank=np.arange(1,len(per_item)+1)/len(per_item)*100;head=int(.2*len(per_item))
    ax.fill_between(rank,per_item,color=COLORS['ngcf'],alpha=.25,lw=0);ax.plot(rank,per_item,color=COLORS['ngcf'],lw=2)
    ax.axvline(20,color=INK,lw=1.5,ls='--')
    ax.annotate('top 20% of movies =\n'+f'{per_item[:head].sum()/per_item.sum():.0%} of ratings',(20,per_item.max()*.75),
                xytext=(8,0),textcoords='offset points',fontsize=9,color=INK,bbox=dict(fc='white',ec='none',pad=1))
    ax.set_xlabel('movies ranked by popularity (%)');ax.set_ylabel('ratings');ax.set_xlim(0,100)
    ax.set_title('Movie popularity long tail',color=INK,loc='left')
    save(fig,fig_dir,'eda.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results')
    ap.add_argument('--split',default='test',choices=['validation','test'])
    ap.add_argument('--data',default='data/processed')
    a=ap.parse_args();main(a.out,a.split);eda(a.data,a.out)
