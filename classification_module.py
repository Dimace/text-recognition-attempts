import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import os.path
import pickle
import shutil
from nltk.corpus import stopwords


with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

with open('idf_coefs', 'rb') as idf:
    coefs = pickle.load(idf)

with open('vocabulary_coefs', 'rb') as idf:
    vocab = pickle.load(idf)

with open('tfidfconverter_trained', 'rb') as tfidfconverter_trained:
    tfidfconverter_trained = pickle.load(tfidfconverter_trained)

print(vocab)

def classify_sites(url_list):
    if(isinstance(url_list, list)):
        parsed_texts = []
        return parsed_texts
    else:
        return []