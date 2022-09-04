import pandas as pd
import os
import sys
import argparse as arg
from tqdm import tqdm 


parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')


args = parser.parse_args(sys.argv[1:])

files = os.listdir(os.path.join('../../Files/', args.dir_path))
files = [file for file in files if file.endswith('.pickle')]

Pro_Vac = pd.DataFrame(columns=['cleanBody', 'score', 'subreddit',  'pred_1',])
Ant_Vac = pd.DataFrame(columns=['cleanBody', 'score', 'subreddit',  'pred_1'])
Neutr = pd.DataFrame(columns=['cleanBody', 'score', 'subreddit', 'pred_1'])

for file in tqdm(files):
    df = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))

    df = df[['cleanBody', 'score', 'subreddit', 'pred_1']]
    df['cleanBody'] = df['cleanBody'].str.replace(r'\[\s', '[', regex=True)
    df['cleanBody'] = df['cleanBody'].str.replace(r'\s\]', ']', regex=True)

    # df['cleanBody'] = df['cleanBody'].str.split(expand=False)

    df0 = df[df['pred_1'] == 0.0]
    df1 = df[df['pred_1'] == 1.0]
    df2 = df[df['pred_1'] == 2.0]

    Ant_Vac = pd.concat([Ant_Vac, df0], ignore_index=True)
    Neutr = pd.concat([Neutr, df1], ignore_index=True)
    Pro_Vac = pd.concat([Pro_Vac, df2], ignore_index=True)

print('done splitting, writing to file')
Ant_Vac.to_parquet(os.path.join('../../Files/', args.dir_path, 'Anti_vacc.parquet'))
Neutr.to_parquet(os.path.join('../../Files/', args.dir_path, 'Neutr_vacc.parquet'))
Pro_Vac.to_parquet(os.path.join('../../Files', args.dir_path, 'Pro_vacc.parquet'))