import os
import pandas as pd 
import argparse as arg
import sys
import datetime
import csv
from itertools import chain

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = arg.ArgumentParser()
parser.add_argument('job', type=str, help='type of job to run, either Merge or Identify')
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('ptype', type=str, help='either Submissions or Comments')

args = parser.parse_args(sys.argv[1:])

# get files in directory and create list of files
def get_files(dir):
    files = []
    for file in os.listdir(dir):
        if file.endswith(".pickle") and os.path.getsize(os.path.join(dir, file)) > 459:
            files.append(os.path.join(dir, file))
    return files.sort()


def identify_missing_files(dir, files):
    missing_files = []
    for file in files:
        if not os.path.exists(os.path.join(dir, file)) or os.path.getsize(os.path.join(dir, file)) <= 459:
            missing_files.append(file)
    return missing_files


def splittimeframe(subreddit, start, end, split): # splits the time into a series of smaller files to pull
    split_list = []
    step = (end - start) / split
    for i in range(split):
            s = int(start + i * step)
            e = int((start + (i + 1) * step) - 1)
            if i == split - 1:
                    e += 86400
            split_list.append(f'{subreddit}-{s}.pickle')

    return split_list

def find_existing_pulls(type, subreddits): #remove existing pulls from subreddits list
    done = os.listdir(f'../../Files/{type}/')
    for i in done:
        done[done.index(i)] = i[:-7]
    res = [i for i in subreddits if i not in done]
    return res

def merge_splits(files, subreddit, ptype): # merges the files down
    merge_candidates = []
    for file in files:
        try:
            if os.path.getsize(f'../../Files/{ptype}/temp/{file}') > 459:
                if file[:-18] == subreddit:
                    merge_candidates.append(f'../../Files/{ptype}/temp/{file}')
        except FileNotFoundError:
            pass

    merge_candidates.sort()
    if len(merge_candidates) == 24:
        df = pd.concat([pd.read_pickle(candidate) for candidate in merge_candidates])
        df.to_pickle(f'../../Files/{ptype}/{subreddit}.pickle')
        print(f'{subreddit} merged')
        for i in merge_candidates:
            os.remove(i)
    else:
        print(f'{subreddit} has only {len(merge_candidates)} files to merge. Merging not possible.')

with open(args.subreddits, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

subreddits = list(chain.from_iterable(subreddits))

if args.job == 'Identify':
    subreddits_not_completed = find_existing_pulls(args.ptype, subreddits)

    start = int(datetime.datetime(2020, 3, 1).timestamp())
    end = int(datetime.datetime(2022, 3, 31).timestamp())
    split = 24
    missing_parts = []
    for subreddit in subreddits_not_completed:
        
        subreddit_parts = splittimeframe(subreddit, start, end, split)
        missing = identify_missing_files(f'../../Files/{args.ptype}/temp/', subreddit_parts)
        for m in missing:
            missing_parts.append(m)


    file = open(f'../../Files/{args.ptype}/temp/missing.csv', "w")
    writer = csv.writer(file, delimiter = "\n")
    for list_ in missing_parts:
        writer.writerow([list_])
    file.close()

if args.job == 'Merge':
    files = []
    files = os.listdir(f'../../Files/{args.ptype}/temp/')
    files.pop(files.index('missing.csv'))

    t = [file[:-18] for file in files]

    subreddits = []
    for i in t:
        if i not in subreddits:
            subreddits.append(i)

    for subreddit in subreddits:
        merge_splits(files, subreddit, args.ptype)
        




