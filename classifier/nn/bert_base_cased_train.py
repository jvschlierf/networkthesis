"""
Trains the Bert Base Cased model using Huggingface
"""


import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset, load_metric
import pandas as pd


train = pd.read_pickle('../../../Files/Submissions/train/train_split_submission.pickle')
train['text'] = train['cleanTitle']
train = train[['text', 'label']]
train_dataset = Dataset.from_pandas(train, preserve_index=False)

valid = pd.read_pickle('../../../Files/Submissions/train/val_split_submission.pickle')
valid['text'] = valid['cleanTitle']
valid = valid[['text', 'label']]
valid_dataset = Dataset.from_pandas(valid, preserve_index=False)

model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



train_dataset_tok = train_dataset.map(tokenize_function, batched=True)
valid_dataset_tok = valid_dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


for name, param in model.named_parameters(): # We train the entire model, not just the classifier
    param.requires_grad = True


metric = load_metric("roc_auc", "multiclass") # we evaluate performance on Area under curve

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits == logits.max(axis=1)[:,None]).astype(int) # We set the highest value to 1, and the rest to 0 to evaluate AUC
    return metric.compute(prediction_scores=predictions, references=labels, multi_class='ovo') # for stability, we choose one vs one compariso (ovo)


training_args = TrainingArguments(
    load_best_model_at_end=True,
    output_dir = '../../../Files/models/bert_base_cased_model/fully_trained/',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=64, 
    evaluation_strategy='epoch',
    logging_dir='../../Files/logs/', 
    save_strategy = "epoch",
    save_steps=10_000, save_total_limit=4, )



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tok,
    eval_dataset=valid_dataset_tok,
    compute_metrics=compute_metrics,
)


trainer.train()
