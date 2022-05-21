import pandas
import os, sys
import argparse

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='File to run through Pipeline')
parser.add_argument('-compress',type=bool, metavar='-c', default=False, help='Skip compression change from .zstd to bzip2' )
parser.add_argument('-outfile',type=str, metavar='-o',default=None, help="Out to different file name")


args = parser.parse_args(sys.argv[1:])

conf = SparkConf().setAppName("Identify_subs").setMaster("local[2]")
sc = SparkContext(conf=conf)

if args.outfile is None:
    args.outfile = args.file

def recompress(): #recompress file if given as .zstd to bzip2

    os.chdir('../../Files/')
    cl = f'zstd -d {args.file}.zst --memory=2048MB -o {args.file} --rm'
    cl2 = f'bzip2 -z {args.file}'
    os.system(cl)
    os.system(cl2)
    print('Recompression done')

if args.compress:
    recompress()

txt = sc.textFile(f"../../Files/{args.file}.bz2")
sqlContext = SQLContext(sc)
df = sqlContext.read.json(txt)

keep = ['author','selftext', 'id', 'name', 'num_comments', 'num_crossposts', 'created_utc', 'crosspost_parent', 'score', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 'title', 'retrieved_utc', 'upvote_ratio']
cols = df.columns
cols = [ele for ele in cols if ele not in keep]
df2 = df.drop(*cols)

df2.write.parquet(f"../../Files/{args.outfile}.parquet", mode='overwrite')