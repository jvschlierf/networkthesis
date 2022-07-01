import pmaw
import os
import pandas as pd
# import multiprocessing
import logging
import argparse as arg
import sys
import datetime
import csv

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
processes = os.cpu_count() - 2

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')


args = parser.parse_args(sys.argv[1:])

start = datetime.datetime(2020, 3, 1)
end = datetime.datetime(2022, 3, 31)


logging.basicConfig(
    level=logging.INFO,
    filename=f'../../Files/logs/pull_subs{datetime.datetime.today()}.log',
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
    for i in subreddit:
        i = i[0]
        logging.info('Pulling subreddit: ' + i)
        start = int(datetime.datetime(2020, 3, 1).timestamp())
        end = int(datetime.datetime(2022, 3, 31).timestamp())
        Subs = pullSubredditSubmissions(i, start, end)
        Comms = pullSubredditComments(i, start, end)
        Subs.to_pickle(f'../../Files/Submissions/{i}.pickle')
        Comms.to_pickle(f'../../Files/Comments/{i}.pickle')


with open(args.subreddits, newline='') as f:
    reader = csv.reader(f)
    subreddits = list(reader)

if __name__ == '__main__':
    main(subreddits) 