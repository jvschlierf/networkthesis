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
files = [file for file in files if file.startswith('d_')]

Pro_Vac = pd.DataFrame(columns=['selftext', 'title', 'score', 'id', 'author', 'subreddit',  'created_utc', 'class_II', 'conf_II'])
# Ant_Vac = pd.DataFrame(columns=['cleanText', 'score', 'subreddit',  'created_utc',  'class_II'])
# Neutr = pd.DataFrame(columns=['cleanText', 'score', 'subreddit', 'created_utc',  'class_II'])

for file in tqdm(files):
    df = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    print(len(df))
    df = df[df['class_I'] == 1.0]
    print(len(df))
    df = df[['selftext', 'title', 'score', 'id', 'author', 'subreddit',  'created_utc', 'class_II', 'conf_II']]
    # df['cleanText'] = df['cleanText'].str.replace(r'\[\s', '[', regex=True)
    # df['cleanText'] = df['cleanText'].str.replace(r'\s\]', ']', regex=True)

    # # df['cleanBody'] = df['cleanBody'].str.split(expand=False)

    # df0 = df[df['class_II'] == 0.0]
    # df1 = df[df['class_II'] == 1.0] 
    # df2 = df[df['class_II'] == 2.0]

    # Ant_Vac = pd.concat([Ant_Vac, df0], ignore_index=True)
    # Neutr = pd.concat([Neutr, df1], ignore_index=True)
    Pro_Vac = pd.concat([Pro_Vac, df], ignore_index=True)

print('done splitting, writing to file')
# Ant_Vac.to_parquet(os.path.join('../../Files/', args.dir_path, 'Anti_vacc_d.parquet'))
# Neutr.to_parquet(os.path.join('../../Files/', args.dir_path, 'Neutr_vacc_d.parquet'))
Pro_Vac.to_pickle(os.path.join('../../Files', args.dir_path, 'ClassifierII_data.pickle'))