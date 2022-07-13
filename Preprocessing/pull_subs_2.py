import pmaw
import os
import pandas as pd
import logging
import argparse as arg
import multiprocessing
import sys
import datetime
import csv
from itertools import chain

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
processes = 64

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('-t', '--type', type=str, help='type information to pull, either subrmissions or comment')

args = parser.parse_args(sys.argv[1:])


logging.basicConfig(
    level=logging.INFO,
    filename=f'../../Files/logs/pull_subs{datetime.datetime.today()}.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def pullSubredditSubmissions(subreddit): # Pulls a subreddit from reddit and saves it to a file
    api = pmaw.PushshiftAPI(jitter='full')
    start = int(datetime.datetime(2020, 3, 1).timestamp())
    end = int(datetime.datetime(2022, 3, 31).timestamp())
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
    logging.info(f'Pulled subreddit {subreddit} number of posts: {len(temp)}')
    temp.to_pickle(f'../../Files/Submissions/{subreddit}.pickle')

def pullSubredditComments(subreddit): # Pulls the comments subreddit from reddit
    api = pmaw.PushshiftAPI(jitter='full')
    start = int(datetime.datetime(2020, 3, 1).timestamp())
    end = int(datetime.datetime(2022, 3, 31).timestamp())  
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
    logging.info(f'Pulled subreddit {subreddit} number of comments: {len(temp)}')
    temp.to_pickle(f'../../Files/Comments/{subreddit}.pickle')

def find_existing_pulls(type, subreddits): #remove existing pulls from subreddits list
    done = os.listdir(f'../../Files/{type}/')
    for i in done:
        done[done.index(i)] = i[:-7]
    res = [i for i in subreddits if i not in done]
    return res


def main(subreddits, type):
    if type == 'Submissions':
       subreddits = find_existing_pulls(type, subreddits)
       logging.info(f'start, pulling {type} for {len(subreddits)} subreddits')
       for subreddit in subreddits:
           pullSubredditSubmissions(subreddit)
    elif type == 'Comments':
        subreddits = find_existing_pulls(type, subreddits)
        logging.info(f'start, pulling {type} for {len(subreddits)} subreddits')
        for subreddit in subreddits:
            pullSubredditComments(subreddit)
    else:
        raise NameError('Please indicate of action to be done')
        quit()

    logging.info('finished')


with open(args.subreddits, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

subreddits = list(chain.from_iterable(subreddits))

if __name__ == '__main__':
   main(subreddits, args.t)