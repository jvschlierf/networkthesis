import pandas as pd
from gensim.models import LdaMulticore, TfidfModel, CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases
from tqdm import tqdm

import time 
import multiprocessing 

columns=['cleanText']

df0 = pd.read_parquet('../../Files/Submissions/score/done/Anti_vac.parquet')
print('loaded first file')
df1 = pd.read_parquet('../../Files/Submissions/score/done/Neutr_vac.parquet')
print('loaded second file')
df2 = pd.read_parquet('../../Files/Submissions/score/done/Pro_vac.parquet')
print('loaded third file, merging')


df = pd.concat([df0, df1, df2], ignore_index=True)
print('merged')
instances = df['cleanText'].to_list()
print('got instances')

phrases = Phrases(instances, min_count=5, threshold=1)
print('got phrases')
instances_colloc = phrases[instances]
print('got instance collocs')
dictionary = Dictionary(instances_colloc)
# get rid of words that are too rare or too frequent
dictionary.filter_extremes(no_below=50, no_above=0.3)
print(dictionary, flush=True)

#replace words by their numerical IDs and their frequency
print("translating corpus to IDs", flush=True)
ldacorpus = [dictionary.doc2bow(text) for text in instances]
# learn TFIDF values from corpus
print("tf-idf transformation", flush=True)
tfidfmodel = TfidfModel(ldacorpus)
# transform raw frequencies into TFIDF
model_corpus = tfidfmodel[ldacorpus]


coherence_values = []



for num_topics in tqdm(range(4, 20)):
    model = LdaMulticore(corpus=model_corpus, 
                         id2word=dictionary, 
                         num_topics=num_topics, random_state=42)

    coherencemodel_umass = CoherenceModel(model=model, 
                                          texts=instances, 
                                          dictionary=dictionary, 
                                          coherence='u_mass')

    coherencemodel_cv = CoherenceModel(model=model, 
                                       texts=instances, 
                                       dictionary=dictionary, 
                                       coherence='c_v')

    umass_score = coherencemodel_umass.get_coherence()
    cv_score = coherencemodel_cv.get_coherence()
    
    print(num_topics, umass_score, cv_score)
    coherence_values.append((num_topics, umass_score, cv_score))



scores = pd.DataFrame(coherence_values, columns=['num_topics', 'UMass', 'CV'])
scores.to_csv('../../Files/Submissions/score/done/topic_scores.csv')
