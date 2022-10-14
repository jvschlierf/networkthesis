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

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    temp = temp[temp['class_I'] == 1.0]
    temp.to_pickle(os.path.join('../../Files', args.dir_path, '/done/', file, 'c_o.pickle'))

print('done writing to file')