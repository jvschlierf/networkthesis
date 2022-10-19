import pandas as pd
import os
import sys
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')
parser.add_argument('pred', choices=('C1', 'C2',  'False'), help='use prediction or not')

args = parser.parse_args(sys.argv[1:])
files = os.listdir(os.path.join('../../Files/', args.dir_path))
files = [file for file in files if file.endswith('.pickle')]


if args.pred == 'C1':
    dictiona = {}
    for file in files:
       
        df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,file))
        filename = file[0:-7]
        print(f'tabulating {filename}')
        dictiona[filename] = [len(df), len(df[df['class_I'] == 0]), len(df[df['class_I'] == 1]), len(df['author'].unique())]
    
    df = pd.DataFrame.from_dict(dictiona, orient='index', columns=['posts', 'Non-Covid', 'Covid', 'authors'])


if args.pred == 'C2':
    files = [file for file in files if file.startswith('d_')]
    dictiona = {}
    for file in files:
       
        df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,file))
        filename = file[2:-7]
        print(f'tabulating {filename}')
        dictiona[filename] = [len(df), len(df[df['class_II'] == 0]), len(df[df['class_II'] == 1]), len(df[df['class_II'] == 2]), len(df['author'].unique())]
    
    df = pd.DataFrame.from_dict(dictiona, orient='index', columns=['posts', 'posts_anti', 'posts_neutral', 'posts_pro', 'authors'])

else:
    print('arg works')
    dictiona = {}
    for filename in os.listdir(os.path.join('../../Files/', args.dir_path)):
        if filename.endswith('.pickle'):
            df = pd.read_pickle(os.path.join('../../Files/', args.dir_path ,filename))
            file = filename[:-7]
            print(f'tabulating {file}')
            dictiona[file] = [len(df), len(df['author'].unique())]
    
    df = pd.DataFrame.from_dict(dictiona, orient='index', columns=['posts', 'authors'])





df.to_csv(os.path.join('../../Files/', args.dir_path, 'EDA_stats.csv'))