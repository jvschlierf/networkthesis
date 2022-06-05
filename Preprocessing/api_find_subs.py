
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


def get_posts(start_epoch, end_epoch, subreddits, outfile, limit=600, repeat=1): # pulls submissions
    api = PushshiftAPI()
    df = pd.DataFrame(columns= ['id', 'url', 'title', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 
                'crosspost_parent', 'crosspost_parent_list', 'num_crossposts', 'created_utc', 'author'])
    
    for j in range(repeat): #for every subreddit, we pull posts given the set criteria
        for i in subreddits:
            gen = api.search_submissions(
                after = start_epoch,
                before = end_epoch,
                subreddit = i,
                filter=['id', 'url', 'title', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 
                'crosspost_parent', 'crosspost_parent_list', 'num_crossposts', 'created_utc', 'author'],
                limit = limit) #limit to avoid going over the rate limit.

            results = list(gen)
            print(f'subreddit: {i}, number of results: {len(results)}')
            temp = pd.DataFrame([thing.d_ for thing in results])
            time.sleep(2) # wait to avoid going over the rate limit. Set higher if limit is increased.
            df = pd.concat([df, temp]) # Add results for one subreddit to the dataframe for all

        start_epoch = df['created_utc'].tail(1).values[0] # Take last post as start for next repeat
        end_epoch = start_epoch + 2678400 #ensuring that end time is one month after start time
    
    df.to_pickle(f'../../Files/{outfile}_temp.pickle')
    return df


def get_crosspost_parent(df, outfile): #find the parents of posts in the observed subreddits using the crosspost_parent_list field
   
    df2 = df[df['crosspost_parent_list'].notna()].reset_index()
    df2 = df2[df2['crosspost_parent_list'].str.len() != 0] #sometimes, this field contains an empty list

    df2['t'] = df2['crosspost_parent_list'].apply(lambda x: dict(x[0])) # Pull crosspost_from information from field 'crosspost_parent_list' (which is in a json format)
    df2['crosspost_from'] = df2['t'].apply(lambda x: x['subreddit'])  # Using the field has the advantage that we can deal with deleted posts which we could not find using the reddit API.
    df2['crosspost_from_id'] = df2['t'].apply(lambda x: x['subreddit_id'])
    df2['crosspost_from_subs'] = df2['t'].apply(lambda x: x['subreddit_subscribers'] )
    df2['crosspost_from_num'] = df2['t'].apply(lambda x: x['num_crossposts'] )
    
    df3 = df2.groupby(['subreddit','subreddit_id','crosspost_from', 'crosspost_from_id']).agg({'subreddit_subscribers': 'mean', 'crosspost_from_subs': 'mean' , 'author' : 'count', 'crosspost_from_num': 'sum'}).reset_index()
    df3.rename(columns = {'author':'count',}, inplace = True)
    df3['crosspost_from_subs'] = df3['crosspost_from_subs'].astype(int)
    
    imp = df3.groupby(['crosspost_from']).agg({'subreddit': 'count', 'count': 'sum'})
    imp = imp.rename(columns ={'count': 'total'}) 
    imp.drop(['subreddit'], axis=1, inplace=True)
    df4 = df.merge(imp, on='crosspost_from', how='left')
    
    df4.to_pickle(f'../../Files/{outfile}_cross_parent.pickle')
    return df4

def get_crosspost_child(df, outfile): #find the crosspost children of posts from the observed subreddits
    api = PushshiftAPI()
    df = df[df['num_crossposts'] > 0] # We do this by looking for posts that have been crossposted
    t = df.groupby(['url']).agg({'num_crossposts': 'sum', 'id':  'max'}).reset_index()
    urls = list(t['url']) # then collect the url of posts / attached links / pictures

    df = pd.DataFrame(columns=['id', 'url', 'title','subreddit', 'subreddit_id', 'subreddit_subscribers',
        'num_crossposts', 'crosspost_parent', 'created_utc', 'author'])
    for i in urls:
        gen2 = api.search_submissions(
        url = i , # and search for them using the pushshift api
        filter=[ 'id', 'url', 'title', 'subreddit', 'subreddit_id', 'subreddit_subscribers',
            'num_crossposts', 'crosspost_parent', 'created_utc', 'author'],
        limit = 200) #limit to avoid going over the rate limit. Can be small, since we're specifically only looking for one link and don't expect too high of a number of posts
        results2 = list(gen2)
        temp = pd.DataFrame([thing.d_ for thing in results2])
        df = pd.concat([df, temp])

    df2 = df[df['num_crossposts'] > 0].reset_index(drop=True) #split into parent posts (number of crossposts > 0 )
    df3 = df[df['num_crossposts'] == 0].reset_index(drop=True) # and child posts (number of crossposts == 0)
    df3 = df3[df3['crosspost_parent'].notna()] # Drop all children where field for parent is empty
    df3['crosspost_parent']  = df3['crosspost_parent'].apply(lambda x: x[3:])
    df2.drop('crosspost_parent', axis=1, inplace=True)
    df2 = df2.rename(columns ={'subreddit':'crosspost_parent','subreddit_id': 'crosspost_parent_id', 'subreddit_subscribers': 'crosspost_parent_subs', 'num_crossposts': 'crosspost_parent_num'})
    df4 = df2.merge(df3, left_on='id', right_on='crosspost_parent', suffixes= ('','_y'))
    df5 = df4.groupby(['subreddit','subreddit_id','crosspost_parent', 'crosspost_parent_id']).agg({'subreddit_subscribers': 'mean', 'crosspost_parent_subs': 'mean' , 'author' : 'count', 'crosspost_parent_num': 'sum'}).reset_index()
    df5.rename(columns = {'author':'count',}, inplace = True)
    df5['crosspost_parent_subs'] = df5['crosspost_parent_subs'].astype(int)

    imp = df5.groupby(['crosspost_from']).agg({'subreddit': 'count', 'count': 'sum'})
    imp = imp.rename(columns ={'count': 'total'}) 
    imp.drop(['subreddit'], axis=1, inplace=True)
    df6 = df.merge(imp, on='crosspost_from', how='left')

    df6.to_pickle(f'../../Files/{outfile}_cross_child.pickle') # save file to avoid straining the API
    return df6

    


def aggregate(df): #aggregate over the subreddits so that we get a list with subreddit - crosspost from and counts
    

    return df3

# def importance(df): #calculate the importance (similar to tf-idf)
#     # OPEN
#     # - make work with Crosspost from
    



def depth():
    pass
    # get list of subreddits_from to then call crosspost_from on them as well

# df = get_crosspost_parent(start_epoch, end_epoch, subreddits, '2021-01-02', repeat=3)

df = pull_crosspost_child(start_epoch, end_epoch, subreddits, '2021-01-02', repeat=3)

# df = pd.read_pickle('../../Files/2021-01-02.pickle')
# df2 = aggregate(df)

# df.rename(columns = {'author':'count',}, inplace = True)
# df3 = importance(df2)
# df3.to_csv('../../Files/tfidf.csv')