import json
import csv
import os

json_obj = {}

with open("text.json", "r") as f:
    old = json.load(f)
    for k, v in old.items():
        json_obj[k] = set(v)



for file in os.listdir('labelled'):
    with open(f'labelled/{file}', 'r') as f:
        l=[]
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader)
        for classified,id,tweet in csv_reader:
            if classified.lower() == 'o':
                json_obj['useless'].add(tweet)
            elif classified.lower() == 'n':
                json_obj['need'].add(tweet)
            elif classified.lower() == 'g':
                json_obj['give'].add(tweet)
            else:
                print('ignoring', id)
                
for k, v in json_obj.items():
    json_obj[k] = list(v)

with open ('text.json', 'w') as f:
    json.dump(json_obj, f)
