import os
import pandas as pd 
import argparse as arg
import sys
import datetime
import csv

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('ptype', type=str, help='either Sumbissions or Comments')

args = parser.parse_args(sys.argv[1:])

print(args.ptype)
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


subreddits_not_completed = find_existing_pulls(args.ptype, args.subreddits)

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



