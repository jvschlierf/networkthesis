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
files = [file for file in files if file.startswith('d_')]

filename = files.pop(0)
df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
df = df[df['class_I']== 1.0]
df['date'] = pd.to_datetime([datetime.fromtimestamp(f) for f in df['created_utc']]).date
df = df.groupby(['date', 'subreddit', 'class_II'], as_index=False).score.count()
df = df.pivot(index='date',columns=['subreddit', 'class_II'])

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    temp = temp[temp['class_I']== 1.0]
    temp['date'] = pd.to_datetime([datetime.fromtimestamp(f) for f in temp['created_utc']]).date
    temp = temp.groupby(['date', 'subreddit','class_II'], as_index=False).score.count()
    temp = temp.pivot(index='date',columns=['subreddit', 'class_II'])
    df = df.add(temp,  fill_value=0)

print('done splitting, writing to file')

df.to_csv(os.path.join('../../Files/', args.dir_path, 'date_sr_c2_db.csv'))