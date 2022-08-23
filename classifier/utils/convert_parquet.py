import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse as arg
import sys

parser = arg.ArgumentParser()
parser.add_argument('dir', type=str, help='directory of files to convert')

args = parser.parse_args(sys.argv[1:])

dirs = args.dir


if __name__ == '__main__':
    print('Converting data...')
    files = os.listdir(os.path.join('../../../Files/',dirs))

    for file in files:
        if file[:-7] != ".pickle":
            files.pop(files.index(file))

    for file in tqdm(files):
        df = pd.read_pickle(os.path.join('../../../Files/',dirs,file)) 
        filename = file[:-7] + '.parquet'
        df.to_parquet(os.path.join('../../../Files/',dirs,filename))
        os.remove(os.path.join('../../../Files/',dirs,file))
    print('Done!')