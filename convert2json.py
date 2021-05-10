import json
import csv
import os
from tqdm import tqdm

data = {'need': set(), 'useless': set(), 'give': set()}


for file in tqdm(os.listdir('labelled')):
    print(file)
    with open(f'labelled/{file}', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader)
        for classified, id, tweet in tqdm(csv_reader):
            if classified.lower() == 'o':
                data['useless'].add(tweet)
            elif classified.lower() == 'n':
                data['need'].add(tweet)
            elif classified.lower() == 'g':
                data['give'].add(tweet)
            else:
                print('Ignoring:', id)

for k, v in data.items():
    data[k] = list(v)

with open('data.json', 'w') as f:
    json.dump(data, f)
