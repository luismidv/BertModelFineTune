import pandas as pd
from transformers import BertModel
from torch import nn
from torch import optim
from datasets import load_dataset
from transformers import AutoTokenizer


#GET THE DATASETS
dataset = pd.read_parquet('./data/train-00000-of-00001.parquet')
print(dataset.head())
#INITIALIZE TOKENIZER

autotokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
#INSTANTIATE THE MODEL
model = BertModel.from_pretrained('bert-base-uncased')

#CREATE CLASSIFIER
classifier = nn.Linear(768, 2)

#CONCATENATE
model = nn.Sequential(model, classifier)

#LOSS FN AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)



result = model(frase)
print(result)