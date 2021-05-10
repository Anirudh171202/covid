from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle as p
import pandas as pd
import json
import random
import numpy as np
import timeit

random.seed(1234)
np.random.seed(1234)

data = []
labels = []
with open("data.json", "r+") as f:
    obj = json.load(f)
    for i in obj.keys():
        data.extend(obj[i])
        labels.extend([i]*(len(obj[i])))

c = list(zip(data, labels))


# Shuffle

random.shuffle(c)
data, labels = zip(*c)

# Get features
vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)
clf = BalancedRandomForestClassifier(n_estimators=150)


X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    train_size=0.75,
    random_state=1234
)

# Train data prep and model
start = timeit.default_timer()
clf.fit(vectorizer.fit_transform(X_train), y_train)
print("Time: ",  str(timeit.default_timer() - start)[:4])

# Accuracy
y_pred = clf.predict(vectorizer.transform(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
reversefactor = dict(zip(["need", "give", "useless"],
                         ["NEED", "GIVE", "USELESS"]))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

print(pd.crosstab(y_test, y_pred, rownames=[
      'Actual'], colnames=['Predicted']))
print()

# Sample predictions
for i in [random.randint(0, len(X_test)) for i in range(5)]:
    print(y_pred[i])
    print(X_test[i].strip())
    print()

# Saving
p.dump(vectorizer, open('./model/vectorizer.pkl', 'wb'))
p.dump(clf, open('./model/clf.pkl', 'wb'))
