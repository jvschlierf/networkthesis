"""
Trains the Bert Base Cased model using Huggingface
"""


import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset, load_metric
import pandas as pd


df = pd.read_pickle('../../../Files/Submissions/train/submission_train_sm.pickle')
df['text'] = df['title']
df['label'] = df['label'].fillna(0)
df['label'] = df['label'].astype(int)
df.loc[df["label"] == -1, "label"] = 2
# df['labels'] = df['label']
df = df[['text', 'label']]
# df = df[df['labels'] != '0']
# df = df[df['labels'].notna()]
# df.loc[df["labels"] == -1, "labels"] = 0


model_name = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = Dataset.from_pandas(df, preserve_index=False)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



tokenized_dataset = dataset.map(tokenize_function, batched=True)

dataset_splitted = tokenized_dataset.shuffle(1337).train_test_split(0.1)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


for name, param in model.named_parameters():
    if name in ['classifier.weight', 'classifier.bias']:
        param.requires_grad = True
    else:
        param.requires_grad = False

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    load_best_model_at_end=True,
    output_dir = '../../../Files/models/bert_base_cased_model/',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32, 
    evaluation_strategy='epoch',
    logging_dir='../../Files/logs/', 
    save_strategy = "epoch",
    save_steps=10_000, save_total_limit=2, )



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_splitted['train'],
    eval_dataset=dataset_splitted['test'],
    compute_metrics=compute_metrics,
)


trainer.train()
