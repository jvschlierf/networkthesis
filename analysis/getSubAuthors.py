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

filename = files.pop(0)
df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
df = df[df['class_I']== 1.0]

df = df.groupby(['author', 'subreddit', 'class_II'], as_index=False).score.count()
df = df.pivot(index='author',columns=['subreddit', 'class_II'])

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    temp = temp[temp['class_I']== 1.0]
    
    temp = temp.groupby(['author', 'subreddit','class_II'], as_index=False).score.count()
    temp = temp.pivot(index='author',columns=['subreddit', 'class_II'])
    df = df.add(temp,  fill_value=0)

print('done splitting, writing to file')

df.to_pickle(os.path.join('../../Files/', args.dir_path, 'author_sub_db.pickle'))