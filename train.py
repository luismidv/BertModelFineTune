import pandas as pd
from transformers import BertModel
from torch import nn
from torch import optim
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.stem.porter import PorterStemmer
def get_data_info(dataset):
    print(dataset.describe())

def identify_null_values(dataset):
    null_values = [col for col in dataset.columns if dataset[col].isnull().any() > 0]
    if len(null_values) > 0:
        print(f"This dataset contains null values")
    else:
        print(f"This dataset doesnt contain any null values")

def stem_words(features):
    features = list(features)
    sentence_counter = 0
    stemmer = PorterStemmer()

def extract_features_labels(dataset):
    features = dataset['text']
    labels = dataset['label']

    features = stem_words(features)

    return features, labels

def tokenize_features(features,autotokenizer):
    tokenized = autotokenizer(features["text"])
    return tokenized



#GET THE DATASETS
dataset = pd.read_parquet('./data/train-00000-of-00001.parquet')

#FUNCTION CALLINGS

    #DATA GENERAL INFORMATION
get_data_info(dataset)

    #NULL VALUES SEARCHING
#identify_null_values(dataset)

    #GET FEATURES AND LABELS
features,labels = extract_features_labels(dataset)



#INITIALIZE TOKENIZER
autotokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

tokenized_features = tokenize_features(dataset, autotokenizer)
print(tokenized_features)

#INSTANTIATE THE MODEL
model = BertModel.from_pretrained('bert-base-uncased')

#CREATE CLASSIFIER
classifier = nn.Linear(768, 2)

#CONCATENATE
model = nn.Sequential(model, classifier)

#LOSS FN AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)


#result = model(frase)
