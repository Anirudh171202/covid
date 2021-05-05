from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd 
import json
import random

# 
# df = None
# with open("text.json", "r") as f:
#     df = pd.DataFrame(json.load(f).items())
# 
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
# 
# data = []
# labels = []
# for i in df.columns.values:
#     data.extend(df[i].tolist())
#     labels.append(i)
# print(data)
# 
data = []
labels = []
with open("text.json", "r+") as f:
    obj = json.load(f)
    for i in obj.keys():
        data.extend(obj[i])
        labels.extend([i]*(len(obj[i])))

features = vectorizer.fit_transform(
    data
)

features_nd = features.toarray()

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        labels,
        train_size=0.80, 
        random_state=1234)

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print(y_pred)
#ACCCURACY
for i in range(0,len(X_test)):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
