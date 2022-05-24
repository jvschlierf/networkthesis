
from psaw import PushshiftAPI
import time
import datetime as dt 
import pandas as pd
import pickle 

api = PushshiftAPI()
start_epoch = int(dt.datetime(2021, 2, 1).timestamp())

# OPEN
# - Args to influence the following:
#   - subreddits as initials (maybe als csv?)
#   - time (start and end)
#   - number of posts
#   - from or to crossposts

subreddits = ['DebateVaccines', 'CovidVaccinated', 'Vaccine', 'Coronavirus', 'LockdownSkepticism', 'HermanCainAward', 'NoNewNormal']

def get_crossposts_from(start_epoch, subreddits, outfile): # pulls submissions, then identifies where the posts are crossposted from
    # OPEN:
    # - Varying time depth
    # - Depth of subreddit -> function to call get_crossposts_from
    # - More than 600 -> get id of last post, set as 'after_id' value


    api = PushshiftAPI()
    results = []
    df = pd.DataFrame(columns= ['subreddit', 'subreddit_id', 'subreddit_subscibers' 
                'crosspost_parent', 'crosspost_parent_list'])

    for i in subreddits: #for every subreddit, we pull posts given the set criteria
        gen = api.search_submissions(
            after = start_epoch,
            subreddit = i,
            filter=['subreddit', 'subreddit_id', 'subreddit_subscribers',
                'crosspost_parent', 'crosspost_parent_list', 'author'],
            limit = 600) #limit to avoid going over the rate limit.

        results = list(gen)
        print(f'subreddit: {i}, number of results: {len(results)}')
        temp = pd.DataFrame([thing.d_ for thing in results])
        time.sleep(4) # wait to avoid going over the rate limit. Set higher if limit is increased.

        df = pd.concat([df, temp]) # Add results for one subreddit to the dataframe for all


    df2 = df[df['crosspost_parent'].notna()].reset_index()

    for i in range(len(df2)): #pull crosspost_from information from field 'crosspost_parent_list' (which is in a json format)
        # Using the field has the advantage that we can deal with deleted posts which we could not find using the reddit API.
        t = dict(df2.loc[i,'crosspost_parent_list'][0])
        df2.loc[i,'crosspost_from'] = t['subreddit']
        df2.loc[i,'crosspost_from_id'] = t['subreddit_id']
        df2.loc[i,'crosspost_from_subs'] = t['subreddit_subscribers'] 

    df2.to_pickle(f'../../Files/{outfile}.pickle') # save file to avoid straining the API
    return df2
    

def get_crossposts_to():
    # OPEN
    # - Define function to find crossposts based on submssion in og subreddit
    # Idea: Filter based on crosspost_num, then search by crosspost_id ?
    pass


def aggregate(df): #aggregate over the subreddits so that we get a list with subreddit - crosspost from and counts
    df3 = df.groupby(['subreddit','subreddit_id','crosspost_from', 'crosspost_from_id']).agg({'subreddit_subscribers': 'mean', 'crosspost_from_subs': 'mean' , 'author' : 'count'}).reset_index()
    df3.rename(columns = {'author':'count',}, inplace = True)
    df3['crosspost_from_subs'] = df3['crosspost_from_subs'].astype(int)

    return df3

def importance(df): #calculate the importance (similar to tf-idf)
    # OPEN
    # - Remerge total counts of subreddits to og df, then calculate tf-idf 
    # - make work with Crosspost from
    imp = df.groupby(['crosspost_from']).agg({'subreddit': 'count', 'count': 'sum'})
    imp.rename(columns ={'count': 'total'}) 
    imp.drop(['subreddit'], axis=1, inplace=True)
    df2 = df.join(imp, on='crosspost_from', how='left')
    return df2


# df = get_crossposts_from(start_epoch, subreddits, 'test')

# df3 = aggregate(df)
# df3.to_csv('../../Files/networktest.csv')

df = pd.read_csv('../../Files/networktest.csv')
df.rename(columns = {'author':'count',}, inplace = True)
df2 = importance(df)
df2.to_csv('../../Files/sfiaf.csv')