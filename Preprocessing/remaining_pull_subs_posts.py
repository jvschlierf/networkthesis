import pmaw
import os
import pandas as pd
import logging
import argparse as arg
from multiprocessing import Pool
import sys
import datetime
import csv
import time
import random
from itertools import chain

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
processes = 6


parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('ptype', type=str, help='type of pull to do')


args = parser.parse_args(sys.argv[1:])


logging.basicConfig(
    level=logging.INFO,
    filename=f'../../Files/logs/remaining_pull_subs{datetime.datetime.today()}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def pullSubreddit(subreddit,start, end, ptype): # Pulls a subreddit from reddit and saves it to a file
    api = pmaw.PushshiftAPI(
    num_workers=10,
    rate_limit=100,
    jitter='full'
    )
    
    if ptype == 'Submissions':
        results = api.search_submissions(
            subreddit=subreddit, 
            after=start, 
            before=end,
            mem_safe=True,
            filter = ('author', 
                'title',
                'created_utc',
                'selftext',
                'url',
                'id',
                'score',
                'num_comments',
                'subreddit',
                'permalink')
            )
    elif ptype == 'Comments':
        results = api.search_comments(
        subreddit=subreddit,
        after=start,
        before=end,
        mem_safe=True,
        filter = ('author',
            'body',
            'created_utc',
            'id',
            'score',
            'permalink',
            'subreddit',
            'parent_id')
        )   

    temp = pd.DataFrame([thing for thing in results])
    logging.info(f'Pulled subreddit {subreddit} number of {ptype}: {len(temp)}')
    return temp
    



def find_existing_pulls(type, subreddits): #remove existing pulls from subreddits list
    done = os.listdir(f'../../Files/{type}/')
    for i in done:
        done[done.index(i)] = i[:-7]
    res = [i for i in subreddits if i not in done]
    return res

def splittimeframe(subreddit, start, end, split): # splits the time into a series of 
    split_list = []
    step = (end - start) / split
    for i in range(split):
            s = int(start + i * step)
            e = int((start + (i + 1) * step) - 1)
            if i == split - 1:
                    e += 86400
            split_list.append([subreddit, s, e])

    return split_list

def main(subreddit): 
    ptype = 'Submissions'
    t = random.randint(0, 600)
    logging.info(f'Pulling subreddit: {subreddit}, but first sleeping for {t}')
    time.sleep(t)
    start = int(datetime.datetime(2020, 3, 1).timestamp())
    end = int(datetime.datetime(2022, 3, 31).timestamp())
    split = 24
    step = (end - start) / split
    temp = pullSubreddit(subreddit[0], subreddit[1], int(subreddit[1])+ int(step), ptype)
    if len(temp) == 0:
        logging.warning(f'Failed to pull Subreddit {subreddit} with {datetime.datetime.fromtimestamp(subreddit[1]).strftime("%m/%d/%Y, %H:%M:%S")}')
        time.sleep(10)
    
    temp.to_pickle(f'../../Files/{ptype}/temp/{subreddit[0]}-{subreddit[1]}.pickle')
    logging.info(f'pulled Subreddit {subreddit} with {ptype} {len(temp)} with begin {datetime.datetime.fromtimestamp(subreddit[1]).strftime("%m/%d/%Y, %H:%M:%S")}')


with open(args.subreddits, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

subreddits = list(chain.from_iterable(subreddits))

subs_start = []
for subreddit in subreddits:
    subs_start.append([subreddit[:-18], int(subreddit[-17:-7])])

if __name__ == '__main__':
    logging.info('start')
    subreddits = find_existing_pulls(args.ptype, subreddits)
    logging.info(f'pulling submissions for {len(subreddits)} subreddits')
    p1 = Pool(processes, maxtasksperchild=2)
    
    p1.map(main, subs_start)
    p1.close()
   
    logging.info('finished')