import csv
import requests

resp = requests.get('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv')

with open('wine_data.csv','wb') as target:
    target.write(resp.text)
