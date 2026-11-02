"""MovieLens 1M -> deterministic chronological leave-two-out implicit-feedback data."""
import argparse
from pathlib import Path
import json
import pandas as pd


def prepare(raw: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(raw / 'ratings.dat', sep='::', engine='python',
                          names=['user_id','item_id','rating','timestamp'], encoding='latin-1')
    users = pd.read_csv(raw / 'users.dat', sep='::', engine='python',
                        names=['user_id','gender','age','occupation','zip'], encoding='latin-1')
    movies = pd.read_csv(raw / 'movies.dat', sep='::', engine='python',
                         names=['item_id','title','genres'], encoding='latin-1')
    # Explicit rating values are ignored: every event is a positive interaction.
    ratings = ratings.drop(columns='rating').drop_duplicates(['user_id','item_id'], keep='last')
    # Fixed-point k-core filtering ensures each retained user/item has >= 5 events.
    while True:
        before = len(ratings)
        uc = ratings.user_id.value_counts()
        ic = ratings.item_id.value_counts()
        ratings = ratings[ratings.user_id.map(uc).ge(5) & ratings.item_id.map(ic).ge(5)].copy()
        if len(ratings) == before:
            break
    uid = sorted(ratings.user_id.unique().tolist())
    iid = sorted(ratings.item_id.unique().tolist())
    u_map = {u: n for n,u in enumerate(uid)}
    i_map = {i: n for n,i in enumerate(iid)}
    ratings['user_idx'] = ratings.user_id.map(u_map)
    ratings['item_idx'] = ratings.item_id.map(i_map)
    ratings = ratings.sort_values(['user_idx','timestamp','item_idx']).reset_index(drop=True)
    pos = ratings.groupby('user_idx').cumcount(ascending=False)
    for name, mask in [('train',pos>=2),('valid',pos==1),('test',pos==0)]:
        ratings.loc[mask, ['user_idx','item_idx','timestamp']].to_csv(out / f'{name}.csv', index=False)
    users['user_idx'] = users.user_id.map(u_map)
    movies['item_idx'] = movies.item_id.map(i_map)
    users[users.user_idx.notna()].to_csv(out/'users.csv',index=False)
    movies[movies.item_idx.notna()].to_csv(out/'movies.csv',index=False)
    pd.DataFrame({'user_id':uid,'user_idx':range(len(uid))}).to_csv(out/'user_map.csv',index=False)
    pd.DataFrame({'item_id':iid,'item_idx':range(len(iid))}).to_csv(out/'item_map.csv',index=False)
    stats = {'users':len(uid),'items':len(iid),'train':int((pos>=2).sum()),
             'valid':int((pos==1).sum()),'test':int((pos==0).sum()),
             'train_mean_history':round(float((pos>=2).sum()/len(uid)),2)}
    (out/'stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path,default=Path('data/raw/ml-1m'))
    parser.add_argument('--out', type=Path,default=Path('data/processed'))
    a=parser.parse_args()
    prepare(a.raw,a.out)
