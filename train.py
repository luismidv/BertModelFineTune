import pandas as pd
import torch
import numpy as np

from transformers import BertModel
from torch import nn
from torch import optim
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.stem.porter import PorterStemmer
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate



def get_data_info(dataset):
    print(dataset.describe())
    labels = dataset["label"]
    print(labels.unique())

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
    new_list = list(features["text"])
    new_list = new_list[:1000]
    counter = 0
    for sentence in new_list:
        tokenized = autotokenizer(sentence, truncation=True)
        new_list[counter] = tokenized
        counter += 1
    return new_list

def tokenize_function(features):
    print(f"Starting tokenization")
    return autotokenizer(features["text"], padding = "max_length", truncation=True)

def get_tensor_input(tokenized_features):
    for dictionary in tokenized_features:
        for key in dictionary.keys():
            dictionary[key] = torch.tensor(dictionary[key])
    return tokenized_features

def input_transformation(model_input):
    for input in model_input:
        keys = input.keys()
        for key in keys:
            input[key] = input[key].view(1, -1)
    return model_input

def model_training(model, input):
    print(input)
    #output = model(**input)

def calculate_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions = predictions, references = labels)

    #GET THE DATASETS
dataset = pd.read_parquet('./data/train-00000-of-00001.parquet')
dataset_hf = Dataset.from_pandas(dataset)

    #FUNCTION CALLINGS

    #DATA GENERAL INFORMATION
get_data_info(dataset)

    #NULL VALUES SEARCHING
#identify_null_values(dataset)

    #GET FEATURES AND LABELS
#tokenized = extract_features_labels(dataset)




    #INITIALIZE TOKENIZER
autotokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

#tokenized_features = tokenize_features(dataset, autotokenizer)
#model_input = get_tensor_input(tokenized_features)
tokenized_features =  dataset_hf.map(tokenize_function, batched=True)

            #INSTANTIATE THE MODEL
#TODO CHECK MODEL INPUTS SIZES SINCE IT SHOULD BE 2 SIZES NOT ONLY ONE
model = BertModel.from_pretrained('bert-base-uncased')
#new_input = input_transformation(model_input)
#model_training(model, new_input)

#output = model(model_input)
#CREATE CLASSIFIER
classifier = nn.Linear(768, 2)

#CONCATENATE
model = nn.Sequential(model, classifier)

#LOSS FN AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 2e-5)

        #CLASS OPTIMIZED FOR TRAINING TRANSFORMERS MODELS FOR CLASSIFICATION, MODEL WITH EXPECTED LABELS (5)
model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-cased', num_labels=5, torch_dtype = "auto")

        #HERE WE HAVE THE HYPERPARAMETERS FOR OUR MODEL
#WE CAN SET EVAL_STRATEGY TO MONITOR EVALUATION METRICS DURING TRAINING
training_args = TrainingArguments(output_dir = "test_trainer", eval_strategy="epoch")
        #METRICS FOR EVALUATIONS
metric = evaluate.load("accuracy")

trainer = Trainer(
    model= model,
    args = training_args,
    train_dataset = tokenized_features,
    eval_dataset = tokenized_features,
    compute_metrics = calculate_metrics

)

trainer.train()



#result = model(frase)
