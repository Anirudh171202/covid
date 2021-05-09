from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import json
import random
import numpy as np
import timeit

data = []
labels = []
with open("text.json", "r+") as f:
    obj = json.load(f)
    for i in obj.keys():
        data.extend(obj[i])
        labels.extend([i]*(len(obj[i])))

c = list(zip(data, labels))

# Preprocessors
vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)


# Shuffle

random.shuffle(c)
data, labels = zip(*c)

# Get features
features = vectorizer.fit_transform(
    data
)

print("Features shape:", features.shape)
features = features.toarray()

X_train_og, X_test_og, y_train, y_test = train_test_split(
    features,
    labels,
    train_size=0.75,
    random_state=1234)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_og)
X_test = scaler.transform(X_test_og)

# Balanced Random Forest

starttime = timeit.default_timer()
clf = BalancedRandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
print("Time: ",  str(timeit.default_timer() - starttime)[:4])

# Accuracy
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
y_pred = clf.predict(X_test)

# Confusion Matrix
reversefactor = dict(zip(["need", "give", "useless"],
                         ["NEED", "GIVE", "USELESS"]))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

print(pd.crosstab(y_test, y_pred, rownames=[
      'Actual'], colnames=['Predicted']))
print()

# Sample predictions
for i in [random.randint(0, len(X_test_og)) for i in range(5)]:
    print(y_pred[i])
    ind = features.tolist().index(X_test_og[i].tolist())
    print(data[ind].strip())
    print()


# Saving
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
