import pandas as pd

import os
import sys
import argparse as arg
from tqdm import tqdm 
from datetime import datetime
import numpy as np

parser = arg.ArgumentParser()
parser.add_argument('dir_path', type=str, help='directory or path file to predict')


args = parser.parse_args(sys.argv[1:])

files = os.listdir(os.path.join('../../Files/', args.dir_path))
files = [file for file in files if file.endswith('.pickle')]

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files', args.dir_path, file))
    results = temp[['label_0', 'label_1', 'label_2']].to_numpy()
    temp_l = np.argmax(results, axis=1)
    conf = results.max(axis=1)
    temp['class_II'] = temp_l
    temp['conf_II'] = conf
    temp.to_pickle(os.path.join('../../Files', args.dir_path, file))