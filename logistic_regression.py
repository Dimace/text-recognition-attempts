# classify review sentiments
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
import os.path
import pickle
import shutil
from nltk.corpus import stopwords

train_data = load_files(r"./train-data")
X, y = train_data.data, train_data.target
original_texts = X[:]
documents = []
print(train_data.target_names)

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    #Remove all number

    document = re.sub(r'[0-9]+', ' ', document)
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=3500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()
idf = tfidfconverter.idf_
print (dict(zip(tfidfconverter.get_feature_names(), idf)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
original_texts_train, original_texts_test = train_test_split(original_texts, test_size=0.4, random_state=0)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print('\nLOGISTIC REGRESSION')
print(classification_report(y_test,y_pred))

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(APP_ROOT, './results/')):
    shutil.rmtree(os.path.join(APP_ROOT, './results/'))

index = 0
for result in y_pred:
    category = train_data.target_names[result]
    dirname = './results/' + category
    if not os.path.exists(os.path.join(APP_ROOT, dirname)):
        os.makedirs(dirname)
    path = os.path.join(APP_ROOT, dirname + '/' + category + '-' + str(index) + '.txt')
    text_file = open(path, "w")
    n = text_file.write(str(original_texts_test[index]))
    text_file.close()
    index = index + 1

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

vocabulary_coefs = dict(zip(tfidfconverter.get_feature_names(), idf))
with open('vocabulary_coefs', 'wb') as picklefile:
    pickle.dump(vocabulary_coefs,picklefile)

with open('tfidfconverter_trained', 'wb') as picklefile:
    pickle.dump(tfidfconverter, picklefile)