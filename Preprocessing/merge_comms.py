import pandas as pd
import os
import argparse as arg
import sys
import datetime
from itertools import chain
import csv
import math

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


parser = arg.ArgumentParser()
parser.add_argument('subreddit', type=str, help='list of subreddits')


args = parser.parse_args(sys.argv[1:])

def identify_missing_files(dir, files):
    missing_files = []
    for file in files:
        if not os.path.exists(os.path.join(dir, file)) or os.path.getsize(os.path.join(dir, file)) <= 459:
            missing_files.append(file)
    return missing_files

def find_existing_pulls(type, subreddits): #remove existing pulls from subreddits list
    done = os.listdir(os.path.join('../../Files/', type))
    for i in done:
        done[done.index(i)] = i[:-7]
    res = [i for i in subreddits if i in done]
    return res


with open(args.subreddit, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

subreddits = list(chain.from_iterable(subreddits))


submissions_done = find_existing_pulls('Submissions/score/', subreddits)

comments_done = find_existing_pulls('Comments/score/', subreddits)

comments_open = [comment for comment in submissions_done if comment not in comments_done]

start = int(datetime.datetime(2020, 3, 1).timestamp())
end = int(datetime.datetime(2022, 3, 31).timestamp())
split = 24
step = (end - start) / split

for comment in comments_open:
    print(f'checking {comment}')
    comment_files = os.listdir(os.path.join('../../Files/Comments/temp'))
    comment_files = [c for c in comment_files if c[:-18] == comment]
    comment_files.sort()
    timestamps = [int(c[-17:-7]) for c in comment_files]
    
    check = 0 
    for i in range(len(timestamps)-1):

        if timestamps[i] == timestamps[i+1] - step:
            check += 1 
    

    
    if check == len(timestamps)- 1:
        print(f'passed length check for {comment}')
        submission = pd.read_pickle(f'../../Files/Submissions/score/{comment}.pickle')
        df1 = pd.read_pickle(os.path.join('../../Files/Comments/temp/', comment_files[0]))
        if math.isclose(submission.created_utc.min(), df1.created_utc.min(), rel_tol=.0000023): # we require the first comment to be created within an hour of the first submission
            print(f'passed start check for {comment}')
            df2 = pd.read_pickle(os.path.join('../../FilesComments/temp/', comment_files[-1]))
            if math.isclose(submission.created_utc.max(), df2.created_utc.max(), rel_tol=0.0004): # we require the last comment to be created within 7 days of the last submission
                print(f'passed end check for {comment}, merging now')
                df = pd.concat([pd.read_pickle(candidate) for candidate in comment_files])
                df.to_pickle(f'../../Files/Comments/{comment}.pickle')
                print(f'{comment} merged')



    
