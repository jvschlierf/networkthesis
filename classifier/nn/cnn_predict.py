import pandas as pd 
from keras.utils import to_categorical
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Dropout, Dense, Activation
import numpy as np


from tensorflow import keras
model = keras.models.load_model('../../../Files/models/CNN8-64.h5')

target = 'label'
input_column = 'cleanText'



label2int = pickle.read('../../../Files/cnnlabel2int.pickle')
# inverted index to translate it back
int2label = pickle.read('../../../Files/cnnint2label.pickle')
word2int = pickle.read('../../../Files/cnnword2int.pickle')

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



files = os.listdir('../../../Files/Submissions/score/')

# remove any files that are not in the .pickle type 
files = [f for f in files if f.endswith('.pickle')]




print(f"setup completed, scoring {len(files)} subreddits")

for file in tqdm(files):
    print(file)
    score = pd.read_pickle(f'../../../Files/Submissions/score/{file}')
    score_instances = score[input_column].apply(str).apply(str.split)
    score_instances_int = convert2ints(score_instances)
    results = model.predict(score_instances_int)
    print(f'predicted for {len(results)} instances')

    for i, j in score.iterrows():
        score.at[i, 'class_I'] = np.argmax(results[i])
        score.at[i, 'conf_I'] = results[i].max()

    score.to_pickle(f'../../../Files/Submissions/score/{file}')
    covid_rel = score['class_I'] == 1
    covid_rel.to_pickle(f'../../../Files/Submissions/score/done/{file}')