# Import Spark NLP
import datetime
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparknlp.base.document_assembler import DocumentAssembler
from sparknlp.base.finisher import Finisher
from sparknlp.annotator.stop_words_cleaner import StopWordsCleaner
from sparknlp.annotator.normalizer import Normalizer
from sparknlp.annotator.token import Tokenizer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer
columns=['cleanText']

spark = SparkSession.builder \
    .appName("Spark NLP")\
    .config("spark.driver.memory","32G")\
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.1.0")\
    .config("spark.kryoserializer.buffer.max", "1000M")\
    .getOrCreate()

NeutralFile = spark.read.parquet("../../Files/Submissions/score/done/Neutr_vacc_d.parquet")
ProFile = spark.read.parquet("../../Files/Submissions/score/done/Pro_vacc_d.parquet")
AntiFile = spark.read.parquet("../../Files/Submissions/score/done/Anti_vacc_d.parquet")




import functools
def unionAll(dfs):
    return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)

Total = unionAll([NeutralFile, ProFile, AntiFile])

sample = Total.sampleBy("class_II", fractions={
    0.0: 0.10,
    1.0: 0.10,
    2.0: 0.10
}, seed=42)


# remove stopwords
document_assembler = DocumentAssembler() \
    .setInputCol("cleanText") \
    .setOutputCol("document") \
    .setCleanupMode("disabled")
# Split sentence to tokens(array)
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
# clean unwanted characters and garbage
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized") \
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

nlp_pipeline = Pipeline(
    stages=[
        document_assembler,
            tokenizer,
            normalizer,
            stopwords_cleaner,  
            finisher])

print('Setup Complete')
# train the pipeline
nlp_model = nlp_pipeline.fit(sample)

# apply the pipeline to transform dataframe.
processed_df  = nlp_model.transform(sample)

tokens_df = processed_df.select('class_II','subreddit', 'score', 'created_utc','tokens')
tokens_df.count()

cv = CountVectorizer(inputCol="tokens", outputCol="features", vocabSize=1000, minDF=3.0)
# train the model
cv_model = cv.fit(tokens_df)
# transform the data. Output column name will be features.
vectorized_tokens = cv_model.transform(tokens_df)

topic_range = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

results = ''
for topic_n in topic_range:
    num_topics = topic_n
    print(datetime.datetime.now(), f"Number of topics: {num_topics}")
    lda = LDA(k=num_topics, maxIter=10)
    model4 = lda.fit(vectorized_tokens)
    ll = model4.logLikelihood(vectorized_tokens)
    lp = model4.logPerplexity(vectorized_tokens)
    results += "*"*25
    results += "\n"
    results += f"Total, NUMBER OF TOPICS: {num_topics}"
    results += "*"*25
    results += "\n"
    results += ("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    results += "\n"
    results +=("The upper bound on perplexity: " + str(lp))
    results += "\n"
    results += "\n"
    
with open("../../Files/models/topic_t_s.txt", "w") as output:
    output.write(results)
