import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset, load_metric
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

trainc = pd.read_pickle('../../../Files/Comments/train/train_split_comments_s.pickle')
trainc['text'] = trainc['cleanBody']
trainc = trainc[['text', 'label']]

trainp = pd.read_pickle('../../../Files/Submissions/train/train_split_submission_r.pickle')
trainp['text'] = trainp['cleanText']
trainp = trainp[['text', 'label']]

# we create a combined dataset and reshuffle it to avoid separation between posts and comments
train = pd.concat([trainc, trainp], axis=0).sample(frac=1).reset_index(drop=True)
train_dataset = Dataset.from_pandas(train, preserve_index=False)

validc = pd.read_pickle('../../../Files/Comments/train/val_split_comments_s.pickle')
validc['text'] = validc['cleanBody']
validc = validc[['text', 'label']]

validp = pd.read_pickle('../../../Files/Submissions/train/val_split_submission_r.pickle')
validp['text'] = validp['cleanText']
validp = validp[['text', 'label']]

valid = pd.concat([validc, validp], axis=0).reset_index(drop=True)

valid_dataset = Dataset.from_pandas(valid, preserve_index=False)

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=511)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_dataset_tok = train_dataset.map(tokenize_function, batched=True)
valid_dataset_tok = valid_dataset.map(tokenize_function, batched=True)

for name, param in model.named_parameters(): # We train the entire model
    param.requires_grad = True


metric = load_metric("roc_auc", "multiclass") # we evaluate performance on ROC Area under curve

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits == logits.max(axis=1)[:,None]).astype(int) # We set the highest value to 1, and the rest to 0 to evaluate AUC
    return metric.compute(prediction_scores=predictions, references=labels, multi_class='ovo') # for stability, we choose one vs one comparison (ovo)


training_args = TrainingArguments(
    load_best_model_at_end=True,
    output_dir = '../../../Files/models/ROBERTA_base/both3/',
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=16, 
    evaluation_strategy='epoch',
    logging_dir='../../../Files/logs/', 
    save_strategy = "epoch",
    save_steps=10_000, save_total_limit=10,
    eval_accumulation_steps=8 )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tok,
    eval_dataset=valid_dataset_tok,
    compute_metrics=compute_metrics,
)

#train the model
trainer.train()
