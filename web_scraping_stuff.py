import requests
import pprint
from bs4 import BeautifulSoup
import json
import os.path

url = 'https://en.wikipedia.org/wiki/Integral'
page = requests.get(url)

soup = (BeautifulSoup(page.content, 'html.parser'))
to_remove = soup.find_all('span')
to_remove.extend(soup.find_all(class_="mwe-math-element"))
for elem in to_remove:
   elem.decompose()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(APP_ROOT, 'draft.txt')
text_file = open(path, "w")
n = text_file.write(soup.get_text())
text_file.close()