import pmaw
import os
import pandas as pd
import logging
import argparse as arg
import sys
from multiprocessing import Pool
import datetime

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
processes = os.cpu_count() - 2

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('-start', type=int, help='start date')
parser.add_argument('-end', type=int, help='end date')


args = parser.parse_args(sys.argv[1:])



# OPEN
# - Args to influence the following:
#   - subreddits as initials (maybe als csv?)
#   - time (start and end)
#   - number of posts
#   - from or to crossposts


logging.basicConfig(
    level=logging.info,
    filename='../../Files/logs/pull_subs.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

def pullSubredditSubmissions(subreddit, start, end): # Pulls a subreddit from reddit and saves it to a file
    api = pmaw.PushshiftAPI(
    
    )
    results = api.search_submissions(
        subreddit=subreddit, 
        after=start, 
        before=end,
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

    )

    results = api.search_comments(
        subreddit=subreddit,
        after=start,
        before=end,
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


def main(subreddit): 
    logging.info('Pulling subreddit: ' + subreddit)
    start = datetime.datetime(2020, 3, 1)
    end = datetime.datetime(2022, 3, 31)
    Subs = pullSubredditSubmissions(subreddit, start, end)
    Comms = pullSubredditComments(subreddit, start, end)
    Subs.to_pickle(f'../../Files/Submissions/{subreddit}.pickle', index=False)
    Comms.to_pickle(f'../../Files/Comments/{subreddit}.pickle', index=False)



if __name__ == '__main__':
    with Pool(os.cpu_count() - 2) as p:
       p.map(main, subreddit) 