import requests
import pprint
from bs4 import BeautifulSoup
import json
import os.path

# improve and finish web scraping process ...

url_sets = [
    {
        'category': 'economics',
        'urls': [
            'https://en.wikipedia.org/wiki/Economic_system',
            'https://en.wikipedia.org/wiki/Economic_ideology',
            'https://en.wikipedia.org/wiki/Economy',
            'https://en.wikipedia.org/wiki/Capitalism',
            'https://en.wikipedia.org/wiki/Communism',
            'https://en.wikipedia.org/wiki/Distributism',
            'https://en.wikipedia.org/wiki/Feudalism',
            'https://en.wikipedia.org/wiki/Inclusive_Democracy',
            'https://en.wikipedia.org/wiki/Market_economy',
            'https://en.wikipedia.org/wiki/Mercantilism'
        ]
    },
    {
        'category': 'mathematics',
        'urls': [
            'https://en.wikipedia.org/wiki/Mathematics',
            'https://en.wikipedia.org/wiki/Geometry',
            'https://en.wikipedia.org/wiki/Probability_theory',
            'https://en.wikipedia.org/wiki/Derivative',
            'https://en.wikipedia.org/wiki/Integral',
            'https://en.wikipedia.org/wiki/Differential_equation',
            'https://en.wikipedia.org/wiki/Function_(mathematics)',
            'https://en.wikipedia.org/wiki/Number_theory',
            'https://en.wikipedia.org/wiki/Algebra',
            'https://en.wikipedia.org/wiki/Mathematical_analysis'
        ]
    },
    {
        'category': 'history',
        'urls': [
            'https://en.wikipedia.org/wiki/History',
            'https://en.wikipedia.org/wiki/Ancient_Greece',
            'https://en.wikipedia.org/wiki/Ancient_Rome',
            'https://en.wikipedia.org/wiki/Roman_Empire',
            'https://en.wikipedia.org/wiki/Roman_Republic',
            'https://en.wikipedia.org/wiki/French_Revolution',
            'https://en.wikipedia.org/wiki/German_revolutions_of_1848%E2%80%931849',
            'https://en.wikipedia.org/wiki/Napoleonic_Wars',
            'https://en.wikipedia.org/wiki/World_War_I',
            'https://en.wikipedia.org/wiki/World_War_II'
        ]
    },
    {
        'category': 'IT',
        'urls': [
            'https://en.wikipedia.org/wiki/Machine_learning',
            'https://en.wikipedia.org/wiki/Artificial_intelligence',
            'https://en.wikipedia.org/wiki/Functional_programming',
            'https://en.wikipedia.org/wiki/Object-oriented_programming',
            'https://en.wikipedia.org/wiki/Class_(computer_programming)',
            'https://en.wikipedia.org/wiki/Object_(computer_science)',
            'https://en.wikipedia.org/wiki/Computer_science',
            'https://en.wikipedia.org/wiki/Data_structure',
            'https://en.wikipedia.org/wiki/Programming_language',
            'https://en.wikipedia.org/wiki/Informatics'
        ]
    }
]

for url_set in url_sets:
    url_set['texts'] = []
    for url in url_set['urls']:
        page = requests.get(url)
        print('page - ' + url + ' is loaded')
        soup = (BeautifulSoup(page.content, 'html.parser'))
        to_remove = soup.find_all('span')
        to_remove.extend(soup.find_all(class_="mwe-math-element"))
        for elem in to_remove:
            elem.decompose()

        paragraphs = soup.find_all('p')
        url_set['texts'].append('')
        index = len(url_set['texts']) - 1

        for paragraph in paragraphs:
            url_set['texts'][index] = url_set['texts'][index] + ' ' + paragraph.get_text()
            
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

for url_set in url_sets:
    category = url_set['category']
    dirname = './train-data/' + category

    if not os.path.exists(os.path.join(APP_ROOT, dirname)):
        os.makedirs(dirname)
    
    for index in range(len(url_set['texts'])):
        file_path = './train-data/' + category + '/' + category  + '-' + str(index) + '.txt'
        path = os.path.join(APP_ROOT, file_path)
        text_file = open(path, "w")
        n = text_file.write(url_set['texts'][index])
        text_file.close()