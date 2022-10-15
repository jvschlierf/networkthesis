import pandas as pd 
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Dropout, Dense, Activation
import numpy as np


import tensorflow as tf
model = tf.keras.models.load_model('../../../Files/models/CNN8-64')

target = 'label'
input_column = 'cleanText'


with open('../../../Files/models/cnnlabel2int.pickle', 'rb') as f:
    label2int = pickle.load(f)
with open('../../../Files/models/cnnint2label.pickle', 'rb') as f:
    int2label = pickle.load(f)
with open('../../../Files/models/cnnword2int.pickle', 'rb') as f:
    word2int= pickle.load(f)

def convert2ints(instances):
    """
    function to apply the mapping to all words
    """
    result = []
    for words in instances:
        # replace words with int, 1 for unknown words
        word_ints = [word2int.get(word, 1) for word in words]
        result.append(word_ints)
    return result



files = ['shitposting.pickle']

# remove any files that are not in the .pickle type 
files = [f for f in files if f.endswith('.pickle')]




print(f"setup completed, scoring {len(files)} subreddits")

for file in tqdm(files):
    print(file)
    score = pd.read_pickle(f'../../../Files/Submissions/score/{file}')
    score_instances = score[input_column].apply(str).apply(str.split)
    score_instances_int = convert2ints(score_instances)
    score_instances_int = pad_sequences(score_instances_int, padding='post', maxlen=78)
    results = model.predict(score_instances_int)
    print(f'predicted for {len(results)} instances')

    class_I = np.argmax(results, axis=1)
    score['class_I'] = class_I
    conf_I = results.max(axis=1)
    score['class_I'] = class_I

    score.to_pickle(f'../../../Files/Submissions/score/{file}')
    covid_rel = score['class_I'] == 1
    covid_rel.to_pickle(f'../../../Files/Submissions/score/done/{file}')