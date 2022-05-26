
from psaw import PushshiftAPI
import time
import datetime as dt 
import pandas as pd
import pickle 
import argparse as arg
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# parser = arg.ArgumentParser()
# parser.add_argument()


api = PushshiftAPI()
start_epoch = int(dt.datetime(2021, 1, 1).timestamp())
end_epoch = int(dt.datetime(2021, 2, 1).timestamp())

# OPEN
# - Args to influence the following:
#   - subreddits as initials (maybe als csv?)
#   - time (start and end)
#   - number of posts
#   - from or to crossposts

subreddits = ['DebateVaccines', 'CovidVaccinated', 'Vaccine', 'Coronavirus', 'LockdownSkepticism', 'HermanCainAward', 'NoNewNormal']

def get_crossposts_from(start_epoch, end_epoch, subreddits, outfile, limit=600, repeat=1): # pulls submissions, then identifies where the posts are crossposted from
    api = PushshiftAPI()
    results = []
    df = pd.DataFrame(columns= ['subreddit', 'subreddit_id', 'subreddit_subscribers', 
                'crosspost_parent', 'crosspost_parent_list', 'created_utc', 'author'])

    for j in range(repeat): #for every subreddit, we pull posts given the set criteria

        for i in subreddits:
            gen = api.search_submissions(
                after = start_epoch,
                before = end_epoch,
                subreddit = i,
                filter=['subreddit', 'subreddit_id', 'subreddit_subscribers',
                    'crosspost_parent', 'crosspost_parent_list', 'created_utc', 'author'],
                limit = limit) #limit to avoid going over the rate limit.

            results = list(gen)
            print(f'subreddit: {i}, number of results: {len(results)}')
            temp = pd.DataFrame([thing.d_ for thing in results])
            time.sleep(2) # wait to avoid going over the rate limit. Set higher if limit is increased.
            df = pd.concat([df, temp]) # Add results for one subreddit to the dataframe for all
            start_epoch = df['created_utc'].tail(1).values[0] # Take last post as start for next repeat
            end_epoch = start_epoch + 2678400 #ensuring that end time is one month after start time

    df2 = df[df['crosspost_parent_list'].notna()].reset_index()
    df2 = df2[df2['crosspost_parent_list'].str.len() != 0] #sometimes, this field contains an empty list

    df2['t'] = df2['crosspost_parent_list'].apply(lambda x: dict(x[0]))
    df2['crosspost_from'] = df2['t'].apply(lambda x: x['subreddit'])
    df2['crosspost_from_id'] = df2['t'].apply(lambda x: x['subreddit_id'])
    df2['crosspost_from_subs'] = df2['t'].apply(lambda x: x['subreddit_subscribers'] )
    df2['crosspost_from_num'] = df2['t'].apply(lambda x: x['num_crossposts'] )

    # for i, r in df2.iterrows(): # Pull crosspost_from information from field 'crosspost_parent_list' (which is in a json format)
    #     # Using the field has the advantage that we can deal with deleted posts which we could not find using the reddit API.
    #     t = dict(r['crosspost_parent_list'][0])
    #     r['crosspost_from'] = t['subreddit']
    #     r['crosspost_from_id'] = t['subreddit_id']
    #     r['crosspost_from_subs'] = t['subreddit_subscribers'] 
    #     r['crosspost_from_num'] = t['num_crossposts'] 

    df2.to_pickle(f'../../Files/{outfile}.pickle') # save file to avoid straining the API
    return df2
    

def get_crossposts_to():
    # OPEN
    # - Define function to find crossposts based on submssion in og subreddit
    # Idea: Filter based on crosspost_num, then search by crosspost_id ?
    pass


def aggregate(df): #aggregate over the subreddits so that we get a list with subreddit - crosspost from and counts
    df3 = df.groupby(['subreddit','subreddit_id','crosspost_from', 'crosspost_from_id']).agg({'subreddit_subscribers': 'mean', 'crosspost_from_subs': 'mean' , 'author' : 'count', 'crosspost_from_num': 'sum'}).reset_index()
    df3.rename(columns = {'author':'count',}, inplace = True)
    df3['crosspost_from_subs'] = df3['crosspost_from_subs'].astype(int)

    return df3

def importance(df): #calculate the importance (similar to tf-idf)
    # OPEN
    # - make work with Crosspost from
    imp = df.groupby(['crosspost_from']).agg({'subreddit': 'count', 'count': 'sum'})
    imp = imp.rename(columns ={'count': 'total'}) 
    imp.drop(['subreddit'], axis=1, inplace=True)
    df = df
    df2 = df.merge(imp, on='crosspost_from', how='left')
    return df2



def depth():
    pass
    # get list of subreddits_from to then call crosspost_from on them as well

df = get_crossposts_from(start_epoch, end_epoch, subreddits, '2021-01-02', repeat=3)

# df = pd.read_pickle('../../Files/2021-01-02.pickle')
df2 = aggregate(df)

# df.rename(columns = {'author':'count',}, inplace = True)
df3 = importance(df2)
df3.to_csv('../../Files/tfidf.csv')