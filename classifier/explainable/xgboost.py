import pandas as pd

import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from joblib import dump

target = 'label'
input_column = 'cleanTitle'

train_data = pd.read_pickle('../../../Files/Submissions/train/train_split_submission.pickle') 
valid_data = pd.read_pickle('../../../Files/Submissions/train/val_split_submission.pickle')
test_data = pd.read_pickle('../../../Files/Submissions/train/test_split_submission.pickle')

train_data = train_data[[target, input_column]]
valid_data = valid_data[[target, input_column]]
test_data = test_data[[target, input_column]]

data = pd.concat([train_data, valid_data, test_data])


train_instances = train_data[input_column].apply(str).apply(str.split)
train_labels = train_data[target]

# collect known word tokens and tags
wordset, labelset = set(), set()

# collect tags from all data, to prevent unseen labels
labelset.update(set(data[target]))

# get the vocabulary
for words in train_instances:
    wordset.update(set(words))

# map words and tags into ints
PAD = '-PAD-'
UNK = '-UNK-'
word2int = {word: i + 2 for i, word in enumerate(sorted(wordset))}
word2int[PAD] = 0  # special token for padding
word2int[UNK] = 1  # special token for unknown words
 
label2int = {label: i for i, label in enumerate(sorted(labelset))}
# inverted index to translate it back
int2label = {i:label for label, i in label2int.items()}


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
                          
train_instances_int = convert2ints(train_instances)
train_labels_int = [label2int[label] for label in train_labels]

test_instances = test_data[input_column].apply(str).apply(str.split)
test_labels = test_data[target]

test_instances_int = convert2ints(test_instances)
test_labels_int = [label2int[label] for label in test_labels]

# convert dev data
val_instances = valid_data[input_column].apply(str).apply(str.split)
val_labels = valid_data[target]

val_instances_int = convert2ints(val_instances)
val_labels_int = [label2int[label] for label in val_labels]

from keras.utils import to_categorical

train_labels_1hot = to_categorical(train_labels_int, len(label2int))
test_labels_1hot = to_categorical(test_labels_int, len(label2int))
val_labels_1hot = to_categorical(val_labels_int, len(label2int))

# compute 95th percentile of training sentence lengths
L = sorted(map(len, train_instances))
MAX_LENGTH = L[int(len(L)*0.95)]
print(MAX_LENGTH)

# apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_instances_int = pad_sequences(train_instances_int, padding='post', maxlen=MAX_LENGTH)
test_instances_int = pad_sequences(test_instances_int, padding='post', maxlen=MAX_LENGTH)
val_instances_int = pad_sequences(val_instances_int, padding='post', maxlen=MAX_LENGTH)


params = { 'mad_depth' : [3, 5, 6, 8, 10],
        'learning_rate' : [0.01, 0.05, 0.1],
        'n_estimators' : [100, 500, 1000],
        'objective': ['multi:softmax', 'multi:softprob'],
        'colsample_bytree': [0.5, 0.7, 1]
}

xgbc = xgb.XGBClassifier(seed = 42)

clf = GridSearchCV(estimator = xgbc,
                param_grid = params,
                scoring='roc_auc_ovo',
                verbose = 1,
                n_jobs = 10
            
)

print('Setup completed, training now')

clf.fit(train_instances_int, train_labels_1hot)

df1 = pd.DataFrame(clf.cv_results)
df1.to_csv('xgboost_results.csv')

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))

model = clf.best_estimator_


dump(clf, 'xgbCVGridSearch.joblib')
dump(model, 'xgbBest_CV.joblib') 