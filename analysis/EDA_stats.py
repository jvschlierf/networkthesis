import pandas as pd
import os
import sys
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')
parser.add_argument('pred', choices=('True', 'False'), help='use predicition or not')

args = parser.parse_args(sys.argv[1:])

if args.pred == 'True':
    dictiona = {}
    for filename in os.listdir(os.path.join('../../Files/', args.dir_path)):
        if filename.endswith('.pickle'):
            df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
            file = filename[2:-7]
            print(f'tabulating {file}')
            dictiona[file] = [len(df), len(df[df['pred_1'] == 0]), len(df[df['pred_1'] == 1]), len(df[df['pred_1'] == 2]), len(df['author'].unique())]
    
    df = pd.DataFrame.from_dict(dictiona, orient='index', columns=['posts', 'posts_pro', 'posts_anti', 'posts_neutral', 'authors'])

else:
    print('arg works')
    dictiona = {}
    for filename in os.listdir(os.path.join('../../Files/', args.dir_path)):
        if filename.endswith('.pickle'):
            df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
            file = filename[2:-7]
            print(f'tabulating {file}')
            dictiona[file] = [len(df), len(df['author'].unique())]
    
    df = pd.DataFrame.from_dict(dictiona, orient='index', columns=['posts', 'authors'])





df.to_csv(os.path.join('../../Files/', args.dir_path, 'EDA_stats.csv'))