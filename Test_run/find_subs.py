import pandas as pd
import os, sys
import argparse

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='File to run through Pipeline')
parser.add_argument('-compress',type=bool, metavar='-c', default=False, help='Skip compression change from .zstd to bzip2' )
parser.add_argument('-outfile',type=str, metavar='-o',default=None, help='Out to different file name')
parser.add_argument('-identify', type=bool, metavar='-i', default=False, help='define subreddits by')
parser.add_argument('-depth', type=int, metavar='-d', default=3, help='Depth of network to be created')
parser.add_argument('-load', type=bool, metavar='-l', default=False, help='Load already recompressed file')


args = parser.parse_args(sys.argv[1:])

conf = SparkConf().setAppName("Identify_subs").setMaster("local[2]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


def recompress(): #recompress file if given as .zstd to bzip2

    os.chdir('../../Files/')
    cl = f'zstd -d {args.file}.zst --memory=2048MB -o {args.file} --rm'
    cl2 = f'bzip2 -z {args.file}'
    os.system(cl)
    os.system(cl2)
    print('Recompression done')

def identify_subs(init_list, frame, depth):
    adjusted_list = []
    network = pd.DataFrame(columns=['subreddit', 'subreddit_id', 'crosspost_parent', 'crosspost_count'])
    for i in init_list:
        temp = sc.sql(f'''SELECT subreddit, subreddit_id, crosspost_parent, COUNT(crosspost_parent) AS crossposted_count
            FROM {frame}
            WHERE subreddit == {i}
            GROUP BY subreddit, subreddit_id, crosspost_parent
            SORT BY COUNT(crosspost_parent) DESC
        ''')
        network.append(temp)

    for j in depth:
        adjusted_list = list(set(network['crosspost_parent']))
        for i in adjusted_list:
            temp = sc.sql(f'''SELECT subreddit, subreddit_id, crosspost_parent, COUNT(crosspost_parent) AS crossposted_count
                FROM {frame}
                WHERE subreddit == {i}
                GROUP BY subreddit, subreddit_id, crosspost_parent
                SORT BY COUNT(crosspost_parent) DESC
            ''')
            
    return network 


if args.outfile is None:
    args.outfile = args.file

if args.compress:
    recompress()


if args.load:
    df =  sqlContext.read.parquet(f"../../Files/{args.file}.parquet")

else:
    txt = sc.textFile(f"../../Files/{args.file}.bz2")
    df = sqlContext.read.json(txt)

keep = ['author','selftext', 'id', 'name', 'num_comments', 'num_crossposts', 'created_utc', 'crosspost_parent', 'score', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 'title', 'retrieved_utc', 'upvote_ratio']
cols = df.columns
cols = [ele for ele in cols if ele not in keep]
df2 = df.drop(*cols)


if args.identify:
    network = identify_subs(init_list, df2, args.depth)







df2.write.parquet(f"../../Files/{args.outfile}.parquet", mode='overwrite')
network.to_csv(f"../../Files/{args.outfile}.csv")