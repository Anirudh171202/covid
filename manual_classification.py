import json

TweetList = []
print("Started Reading Tweet")
with open('text.json') as f:
    for jsonObj in f:
        Dict = json.loads(jsonObj)
        TweetList.append(Dict)

print("Printing each JSON Decoded Object")
for tweet in TweetList:
    print(tweet["tweet"])
    intent = int(input())


