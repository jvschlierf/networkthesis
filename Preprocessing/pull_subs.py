import pmaw
import os
import pandas as pd
import logging
import argparse as arg

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

parser = arg.ArgumentParser()
parser.add_argument('subreddits', type=str, help='csv list of subreddits to pull')
parser.add_argument('-start', type=int, help='start date')
parser.add_argument('-end', type=int, help='end date')


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