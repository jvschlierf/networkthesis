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
from itertools import chain

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
processes = 64 

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')


args = parser.parse_args(sys.argv[1:])


logging.basicConfig(
    level=logging.INFO,
    filename=f'../../Files/logs/pull_subs{datetime.datetime.today()}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def pullSubredditSubmissions(subreddit, start, end): # Pulls a subreddit from reddit and saves it to a file
    api = pmaw.PushshiftAPI(
    jitter='full'
    )
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

    temp = pd.DataFrame([thing for thing in results])
    return temp

def pullSubredditComments(subreddit, start, end): # Pulls the comments subreddit from reddit
    api = pmaw.PushshiftAPI(
    jitter='full'
    )

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
    return temp


def find_existing_pulls(type, subreddits): #remove existing pulls from subreddits list
    done = os.listdir(f'../../Files/{type}/')
    for i in done:
        done[done.index(i)] = i[:-7]
    res = [i for i in subreddits if i not in done]
    return res

def main(subreddit): 
    logging.info('Pulling subreddit: ' + subreddit)
    start = int(datetime.datetime(2020, 3, 1).timestamp())
    end = int(datetime.datetime(2022, 3, 31).timestamp())
    # Subs = pullSubredditSubmissions(subreddit, start, end)
    # Subs.to_pickle(f'../../Files/Submissions/{subreddit}.pickle')
    # logging.info(f'Pulled subreddit {subreddit} number of posts: {len(Subs)}')
    
    Comms = pullSubredditComments(subreddit, start, end)
    Comms.to_pickle(f'../../Files/Comments/{subreddit}.pickle')
    logging.info(f'Pulled subreddit {subreddit} number of comments: {len(Comms)}')


with open(args.subreddits, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

subreddits = list(chain.from_iterable(subreddits))

if __name__ == '__main__':
    logging.info('start')
    subreddits = find_existing_pulls('Comments', subreddits)
    logging.info(f'pulling comments for {len(subreddits)} subreddits')
    p1 = Pool(int(processes/4), maxtasksperchild=2)
    p2 = Pool(int(processes/4), maxtasksperchild=2)
    p3 = Pool(int(processes/4), maxtasksperchild=2)
    p4 = Pool(int(processes/4), maxtasksperchild=2)
    subs_1 = subreddits[0:len(subreddits)/4]
    
    p1.map(main, subs_1)
    logging.info(f'started first round of subs with {len(subs_1)}')

    subs_2 = subreddits[len(subreddits)/4:len(subreddits)/2]
    time.sleep(20)
    p2.map(main, subs_1)
    logging.info(f'started second round of subs with {len(subs_2)}')

    subs_3 = subreddits[len(subreddits)/2: (len(subreddits)/4)*3]
    time.sleep(20)
    p3.map(main, subs_3)
    logging.info(f'started third round of subs with {len(subs_3)}')

    subs_4 = subreddits[(len(subreddits)/4)*3:]
    time.sleep(20)
    p4.map(main, subs_4)
    logging.info(f'started third round of subs with {len(subs_4)}')
    

    p1.close()
    p2.close()
    p3.close()
    p4.close()
    logging.info('finished')