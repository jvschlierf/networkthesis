import pandas as pd

import os
import sys
import argparse as arg
from tqdm import tqdm 
from datetime import datetime

parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')


args = parser.parse_args(sys.argv[1:])

files = os.listdir(os.path.join('../../Files/', args.dir_path))
files = [file for file in files if file.endswith('.pickle')]

filename = files.pop(0)
df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
df = df.groupby(['subreddit', 'class_I'], as_index=False).score.count()
df = df.pivot(index='subreddit',columns='class_I')
df.columns = df.columns.droplevel(0)
df = df.reset_index()
df.rename_axis('index', axis=1, inplace=True)
df.rename(columns={0.0: 'Non-Covid', 1.0: 'Covid'}, inplace=True)

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    temp = temp.groupby(['subreddit', 'class_I'], as_index=False).score.count()
    temp = temp.pivot(index='subreddit',columns='class_I')
    temp.columns = temp.columns.droplevel(0)
    temp = temp.reset_index()
    temp.rename_axis('index', axis=1, inplace=True)
    temp.rename(columns={0.0: 'Non-Covid', 1.0: 'Covid'}, inplace=True)
    df = df.add(temp,  fill_value=0)

print('done splitting, writing to file')

df.to_csv(os.path.join('../../Files/', args.dir_path, 'c1_distr_db.csv'))