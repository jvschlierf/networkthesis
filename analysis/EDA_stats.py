import pandas as pd
import os


length = {}
post_pro = {}
post_neut = {}
post_anti = {}
posters = {}

for filename in os.listdir('../../Files/Submissions/score/done/'):
    if filename.endswith('.pickle'):
        df = pd.read_pickle('../../Files/Submissions/score/done/' + filename)
        file = filename[2:-7]
        print(f'tabulating {file}')
        length[file] = len(df)
        post_pro[file] = len(df[df['pred_1'] == 0])
        post_neut[file] = len(df[df['pred_1'] == 1])
        post_anti[file] = len(df[df['pred_1'] == 2])
        posters[file] = len(df['author'].unique())


df = pd.DataFrame.from_dict(length)
df['post_pro'] = df.map(post_pro)
df['post_neut'] = df.map(post_neut)
df['post_anti'] = df.map(post_anti)
df['posters'] = df.map(posters)
df.to_csv('../../Files/Submissions/score/EDA_stats.csv')