from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import json
import random
from numpy import mean, std
import numpy as np
import timeit

# DATA MANIPULATION
vectorizer = CountVectorizer(
    analyzer='word',
    lowercase=False,
)

data = []
labels = []
with open("text.json", "r+") as f:
    obj = json.load(f)
    for i in obj.keys():
        data.extend(obj[i])
        labels.extend([i]*(len(obj[i])))

# SHUFFLE

c = list(zip(data, labels))

random.shuffle(c)
data, labels = zip(*c)

# PREPROCESSING
features = vectorizer.fit_transform(
    data
)

features_nd = features.toarray()

X_train_og, X_test_og, y_train, y_test = train_test_split(
    features_nd,
    labels,
    train_size=0.80,
    random_state=1234)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_og)
X_test = scaler.transform(X_test_og)

# LOGISTIC REGRESSION (85% accuracy)
starttime = timeit.default_timer()
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Accuracy for logistic : ", accuracy_score(y_test, y_pred),
      "\ntime taken ",  str(timeit.default_timer() - starttime)[:4],  "s\n")

# RANDOM FOREST (90% accuracy)
starttime = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ACCURACY FOR LOGISTIC/RANDOM FOREST
print("Accuracy for random forest : ", accuracy_score(y_test, y_pred),
      "\ntime taken : ",  str(timeit.default_timer() - starttime)[:4], "s\n")

y_pred = clf.predict(X_test)

# NICE TABLE FOR SEEING TESTING RESULTS
reversefactor = dict(zip(["need", "give", "useless"],
                         ["NEED", "GIVE", "USELESS"]))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix

print(pd.crosstab(y_test, y_pred, rownames=[
      'Actual Classification'], colnames=['Predicted CLASSIFICATION']))
print()
# #ADABOOST (75% accuracy)
# starttime = timeit.default_timer()
# model = AdaBoostClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
#
# ACCURACY FOR ADABOOST
# print('Accuracy for adaboost: %.3f (%.3f)' % (mean(n_scores), std(n_scores)), "\ntime taken : ", str(timeit.default_timer() - starttime)[:5], "s\n")

# CODE FOR PRINTING INPUTS WITH OUTPUTS FOR DEBUGGING
for i in [random.randint(0, len(X_test_og)) for i in range(5)]:
    print(y_pred[i])
    ind = features_nd.tolist().index(X_test_og[i].tolist())
    print(data[ind].strip())
    print()
