import json
import csv
import os

json_obj = {
    'give' : [],
    'need' : [],
    'useless' : []
}

for file in os.listdir('labelled'):
    with open(f'labelled/{file}', 'r') as f:
        l=[]
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader)
        for classified,id,tweet in csv_reader:
            if classified.lower() == 'o':
                json_obj['useless'].append(tweet)
            elif classified.lower() == 'n':
                json_obj['need'].append(tweet)
            elif classified.lower() == 'g':
                json_obj['give'].append(tweet)
            else:
                print('ignoring', id)

with open ('text.json', 'w') as f:
    json.dump(json_obj, f)