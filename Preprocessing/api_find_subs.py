
from pmaw import PushshiftAPI
import datetime as dt 
import pandas as pd
import argparse as arg
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# parser = arg.ArgumentParser()
# parser.add_argument()


# api = PushshiftAPI()


# OPEN
# - Args to influence the following:
#   - subreddits as initials (maybe als csv?)
#   - time (start and end)
#   - number of posts
#   - from or to crossposts



def get_posts(start_epoch, end_epoch, subreddits, limit=1000, repeat=1): # pulls submissions
    api = PushshiftAPI()
    df = pd.DataFrame(columns= ['id', 'url', 'title', 'subreddit', 'selftext', 'subreddit_subscribers', 
                'crosspost_parent', 'crosspost_parent_list', 'num_crossposts', 'created_utc', 'author'])
    
    
    for j in range(repeat): #for every subreddit, we pull posts given the set criteria
        for i in subreddits:
            results = api.search_submissions(
                after = start_epoch,
                before = end_epoch,
                subreddit = i,
                filter=['id', 'url', 'title', 'subreddit', 'selftext', 'subreddit_subscribers', 
                'crosspost_parent', 'crosspost_parent_list', 'num_crossposts', 'created_utc', 'author'],
                limit = limit) #limit to avoid going over the rate limit.
            
            temp = pd.DataFrame([thing for thing in results])
            df = pd.concat([df, temp]) # Add results for one subreddit to the dataframe for all

    start_epoch = df['created_utc'].tail(1).values[0] # Take last post as start for next repeat
    end_epoch = start_epoch + 2678400 #ensuring that end time is one month after start time
    print(f'repeat: {j} pulled {len(temp)} items from {len(subreddits)} subreddits')
    
    return df


def aggregate(df): #aggregate over the subreddits so that we get a list with subreddit - crosspost from and counts
    
    df2 = df.groupby(['subreddit','crosspost_parent']).agg({'subreddit_subscribers': 'mean', 'crosspost_parent_subs': 'mean' , 'author' : 'count', 'crosspost_parent_num': 'sum'}).reset_index()
    df2.rename(columns = {'author':'count',}, inplace = True)
    df2['crosspost_parent_subs'] = df2['crosspost_parent_subs'].astype(int)

    imp = df2.groupby(['crosspost_parent']).agg({'subreddit': 'count', 'count': 'sum'})
    imp = imp.rename(columns ={'count': 'total'}) 
    imp.drop(['subreddit'], axis=1, inplace=True)
    df3 = df2.merge(imp, on='crosspost_parent', how='left')

    print('aggregated')
    return df3


def get_crosspost_parent(df, outfile, depth_lim): #find the parents of posts in the observed subreddits using the crosspost_parent_list field
   
    df2 = df[df['crosspost_parent_list'].notna()].reset_index()
    df2 = df2[df2['crosspost_parent_list'].str.len() != 0] #sometimes, this field contains an empty list

    df2['t'] = df2['crosspost_parent_list'].apply(lambda x: dict(x[0])) # Pull crosspost_from information from field 'crosspost_parent_list' (which is in a json format)
    df2['crosspost_parent'] = df2['t'].apply(lambda x: x['subreddit'])  # Using the field has the advantage that we can deal with deleted posts which we could not find using the reddit API.
    df2['crosspost_parent_subs'] = df2['t'].apply(lambda x: x['subreddit_subscribers'] )
    df2['crosspost_parent_num'] = df2['t'].apply(lambda x: x['num_crossposts'] )
    
    df2 = df2[(df2['crosspost_parent'].str.startswith('u_') == False) & (df2['subreddit'].str.startswith('u_') == False)] # get rid off users as subreddit
    df3 = aggregate(df2)
    print('identified parents')
    df3.to_pickle(f'../../Files/{outfile}_cross_parent_{depth_lim}.pickle')
    return df3

def get_crosspost_child(df, outfile, depth_lim): #find the crosspost children of posts from the observed subreddits
    api = PushshiftAPI()
    df = df[df['num_crossposts'] > 0] # We do this by looking for posts that have been crossposted
    t = df.groupby(['url']).agg({'num_crossposts': 'sum', 'id':  'max'}).reset_index()
    urls = list(t['url']) # then collect the url of posts / attached links / pictures

    df = pd.DataFrame(columns=['id', 'url', 'title', 'subreddit', 'selftext', 'subreddit_subscribers',
        'num_crossposts', 'crosspost_parent', 'created_utc', 'author'])

    for j in urls:
        results2 = api.search_submissions(
            url = j, # and search for them using the pushshift api
            filter=[ 'id', 'url', 'title', 'subreddit', 'selftext', 'subreddit_subscribers',
                'num_crossposts', 'crosspost_parent', 'created_utc', 'author'],
            limit = 100) #limit to avoid going over the rate limit. Can be small, since we're specifically only looking for one link and don't expect too high of a number of posts
    
        temp = pd.DataFrame([thing for thing in results2])
        df = pd.concat([df, temp])

    df.to_pickle(f'../../Files/{outfile}_raw_child_{depth_lim}.pickle')
    df2 = df[df['num_crossposts'] > 0].reset_index(drop=True) #split into parent posts (number of crossposts > 0 )
    df3 = df[df['num_crossposts'] == 0].reset_index(drop=True) # and child posts (number of crossposts == 0)
    df3 = df3[df3['crosspost_parent'].notna()] # Drop all children where field for parent is empty
    df3['crosspost_parent']  = df3['crosspost_parent'].apply(lambda x: x[3:])
    df2.drop('crosspost_parent', axis=1, inplace=True)
    df2 = df2.rename(columns ={'subreddit':'crosspost_parent', 'subreddit_subscribers': 'crosspost_parent_subs', 'num_crossposts': 'crosspost_parent_num'})
    df4 = df2.merge(df3, left_on='id', right_on='crosspost_parent', suffixes= ('','_y'))

    df4 = df4[(df4['crosspost_parent'].str.startswith('u_') == False) & (df4['subreddit'].str.startswith('u_') == False)] # get rid off users as subreddit
    df5 = aggregate(df4)
    
    df
    print('identified children')
    df5.to_pickle(f'../../Files/{outfile}_cross_child_{depth_lim}.pickle') # save file to avoid straining the API
    return df5

def update_seed_subs(df, subreddits): #find the subreddits to look for in the next 
    t = df['subreddit'].drop_duplicates().to_list()
    t.extend(df['crosspost_parent'].drop_duplicates().to_list())
    
    res = []
    for i in t:
        if i not in res and i not in subreddits:
            res.append(i)

    print(f'updated subreddit list, new list contains {len(res)} items')
    return res


def crosspost_ratio(df):
    df2 = df.groupby('subreddit').agg({'crosspost_parent': 'count', 'num_crossposts': lambda x: np.count_nonzero(x), 'subreddit_subscribers' : 'count'}).reset_index()
    df3 = df2.rename(columns ={'subreddit_subscribers':'total', 'num_crossposts': 'crosspost_child'})
    df3['crosspost_parent_%'] = df3['crosspost_parent'] / df3['total']
    df3['crosspost_child_%'] = df3['crosspost_child'] / df3['total']
    return df3
    

def depth(start_epoch, end_epoch, subreddits, outfile, limit=600, repeat=1, depth_lim=1):
    outdf = pd.DataFrame(columns=['subreddit', 'crosspost_parent', 'subreddit_subscribers', 'crosspost_parent_subs', 'count', 'crosspost_parent_num', 'total'])
    total_subs = []
    while depth_lim > 1:
        df = get_posts(start_epoch, end_epoch, subreddits, outfile, limit, repeat)
        print(f'pulled {len(subreddits)} subreddits, number of results: {len(df)}')
        df.to_pickle(f'../../Files/{outfile}_raw{depth_lim}.pickle')
        ratio = crosspost_ratio(df)
        ratio.to_pickle(f'../../Files/{outfile}_ratio_temp{depth_lim}.pickle')

        df2 = get_crosspost_parent(df, outfile, depth_lim)
        outdf = outdf.append(df2)
        
        df3 = get_crosspost_child(df, outfile, depth_lim)
        outdf = outdf.append(df3)
        
        total_subs.extend(subreddits)
        subreddits = update_seed_subs(outdf, subreddits)
        
        depth_lim -= 1
        outdf.to_pickle(f'../../Files/{outfile}_cross_temp_{depth_lim}.pickle')
        print(f'now getting {len(subreddits)} new subreddits, depth is {depth_lim}')
    

    outdf.to_pickle(f'../../Files/{outfile}_cross.pickle')

    total_subs.extend(subreddits)
    np.savetxt("../../Files/subs.csv", 
           total_subs,
           delimiter =", ", 
           fmt ='% s')



start_epoch = int(dt.datetime(2021, 1, 1).timestamp())
end_epoch = int(dt.datetime(2021, 2, 1).timestamp())

subreddits = ['DebateVaccines', 'Vaccine', 'Coronavirus', 'COVID19',  'LockdownSkepticism', 'NoNewNormal', 'HermanCainAward', 'CovidVaccinated']

depth(start_epoch, end_epoch, subreddits, 'test0609', limit=2000, repeat=8, depth_lim=3)


# df = pd.read_pickle('../../Files/2021-01-02.pickle')
# df2 = aggregate(df)

# df.rename(columns = {'author':'count',}, inplace = True)
# df3 = importance(df2)
# df3.to_csv('../../Files/tfidf.csv')