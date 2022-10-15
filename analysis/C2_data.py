import pandas as pd

import os
import sys
from tqdm import tqdm 


files = os.listdir(os.path.join('../../Files/Submissions/score/'))
files = [file for file in files if file.endswith('.pickle')]

for file in tqdm(files):
    temp = pd.read_pickle(os.path.join('../../Files/Submissions/score/', file))
    temp = temp[temp['class_I'] == 1.0]
    temp.to_pickle(os.path.join('../../Files/Submissions/score/done/', file))
print('done writing to file')