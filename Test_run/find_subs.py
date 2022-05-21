import pandas
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
conf = SparkConf().setAppName("Identify_subs").setMaster("local[10]")
sc = SparkContext(conf=conf)

txt = sc.textFile("../../Files/sample.bz2", use_unicode=False)
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
df = sqlContext.read.json(txt)

keep = ['author','selftext', 'id', 'name', 'num_comments', 'num_crossposts', 'created_utc', 'crosspost_parent', 'score', 'subreddit', 'subreddit_id', 'subreddit_subscribers', 'title', 'retrieved_utc', 'upvote_ratio']

cols = df.columns
cols = [ele for ele in cols if ele not in keep]
df2 = df.drop(*cols)