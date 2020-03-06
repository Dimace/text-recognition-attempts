import re
import pickle
from bs4 import BeautifulSoup
from sklearn.datasets import load_files
import requests
from nltk.stem import WordNetLemmatizer

categories = ['IT', 'economics', 'history', 'mathematics']

with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

with open('tfidfconverter_trained', 'rb') as tfidfconverter_trained:
    tfidfconverter_trained = pickle.load(tfidfconverter_trained)

def classify_sites(url_list):
    if(isinstance(url_list, list)):
        parsed_texts = []
        raw_texts = fetch_texts_by_url(url_list)
        processed_texts = process_texts(raw_texts)
        samples = dict(zip(url_list, processed_texts))
        transformed_texts = tfidfconverter_trained.transform(processed_texts).toarray()

        prediction_categories = map(lambda index: categories[index], model.predict(transformed_texts))
        result = dict(zip(url_list, prediction_categories))
        return result
    else:
        return []

def fetch_texts_by_url(url_set):
    result = []
    for url in url_set:
        page = requests.get(url)
        print('page - ' + url + ' is loaded')
        soup = (BeautifulSoup(page.content, 'html.parser'))
        to_remove = soup.find_all('span')
        to_remove.extend(soup.find_all(class_="mwe-math-element"))
        for elem in to_remove:
            elem.decompose()

        paragraphs = soup.find_all('p')
        text = ''

        for paragraph in paragraphs:
            if(len(text) == 0):
                text = paragraph.get_text()
            else:
                text = text + ' ' + paragraph.get_text()
        result.append(text)
    
    return result

def process_texts(texts):
    result = []
    stemmer = WordNetLemmatizer()

    for text in texts:

        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(text))

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
        
        result.append(document)

    return result

def test_module():
    test_set = [
        'https://en.wikipedia.org/wiki/Fascism',
        'https://en.wikipedia.org/wiki/Neoliberalism',
        'https://en.wikipedia.org/wiki/Compiler',
        'https://en.wikipedia.org/wiki/Euclidean_geometry',
        'https://en.wikipedia.org/wiki/Inflation',
        'https://en.wikipedia.org/wiki/Battle_of_Britain',
        'https://en.wikipedia.org/wiki/Source_code',
        'https://en.wikipedia.org/wiki/Algorithm'
    ]

    result = classify_sites(test_set)

    print('TESTING CLASSIFICATION MODULE...')
    print(result)
    return None

if __name__ == "__main__": 
    test_module()
