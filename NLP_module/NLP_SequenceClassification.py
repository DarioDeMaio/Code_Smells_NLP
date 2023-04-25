import data_loader as dl
import pandas as pd
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf

dataframe = dl.load_data()
print(dataframe)

# Carica il tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Carica il modello
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
