"""
File to control text classification models as part of Analysis of Pro- & Anti- Vaccine behavior on reddit
"""
import logging
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", model_args={"num_labels": 3},  )

inputs = tokenizer("Hello, my dog is ugly", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])
