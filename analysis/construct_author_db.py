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

filename = files.pop(0)
df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
        
sl = df.groupby(['author', 'pred_1'], as_index=False).num_comments.count()
df = sl.pivot(index='author',columns='pred_1')


for filename in tqdm(files):
    if filename.endswith('.pickle'):
        temp = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
        
        sl = temp.groupby(['author', 'pred_1'], as_index=False).num_comments.count()
        temp = sl.pivot(index='author',columns='pred_1')

        df = df.add(temp,  fill_value=0)
        



df.to_csv(os.path.join('../../Files/', args.dir_path, 'author_db.csv'))