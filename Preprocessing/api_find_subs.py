
from psaw import PushshiftAPI
import time
import datetime as dt 
import pandas as pd 

api = PushshiftAPI()
start_epoch = int(dt.datetime(2021, 2, 1).timestamp())


subreddits = ['DebateVaccines', 'CovidVaccinated', 'Vaccine', 'Coronavirus', 'LockdownSkepticism', 'HermanCainAward', 'NoNewNormal']

def get_crossposts(start_epoch, subreddits): #Need to define varying time / depth of crosspost to look at
    
    api = PushshiftAPI()
    results = []
    df = pd.DataFrame(columns= ['subreddit', 'subreddit_id',  
                'crosspost_parent', 'crosspost_parent_list'])

    for i in subreddits:
        gen = api.search_submissions(
            after = start_epoch,
            subreddit = i,
            filter=['subreddit', 'subreddit_id',  
                'crosspost_parent', 'crosspost_parent_list'],
            limit = 600)

        results = (list(gen))
        print(f'subreddit: {i}, number of results: {len(results)}')
        temp = pd.DataFrame([thing.d_ for thing in results])
        time.sleep(2)

        df = pd.concat([df, temp])


    df2 = df[df['crosspost_parent'].notna()].reset_index()

    for i in range(len(df2)):
        t = dict(df2.loc[i,'crosspost_parent_list'][0])
        df2.loc[i,'crosspost_from'] = t['subreddit']
        df2.loc[i,'crosspost_from_id'] = t['subreddit_id']
        # df2.loc[i,'crosspost_from_subs'] = t['subreddit_subscribers'] # find way to get average of subscribers so groupby works

    df3 = pd.DataFrame(df2.groupby(['subreddit','subreddit_id','crosspost_from', 'crosspost_from_id']).size().reset_index())
    df3.rename(columns = {0:'Count'}, inplace = True)
    # df3['crosspost_from_subs'] = df3['crosspost_from_subs'].astype(int)
    
    return df3



df3 = get_crossposts(start_epoch, subreddits)

df3.to_csv('../../Files/networktest.csv')