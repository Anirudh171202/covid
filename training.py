from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler
from nltk import PorterStemmer
import gensim
import pickle
import pandas as pd
import json
import re
import random
import numpy as np
import timeit

# Load Data

data = []
labels = []
with open("text.json", "r+") as f:
    obj = json.load(f)
    for i in obj.keys():
        data.extend(obj[i])
        labels.extend([i]*(len(obj[i])))

c = list(zip(data, labels))

random.shuffle(c)
data, labels = zip(*c)

# Data Preprocessing


# Clean tweets: rm twitter handles, punctuations, special characters

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


data_tidy = pd.Series(np.vectorize(remove_pattern)(pd.Series(data), "@[\w]*"))
data_tidy = data_tidy.str.replace("[^a-zA-Z#]", " ")
tokenized = data_tidy.apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized = tokenized.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming

model_w2v = gensim.models.Word2Vec(
    tokenized,
    vector_size=5,  # desired no. of features/independent variables
    window=5,  # context window size
    min_count=2,
    sg=1,  # 1 for skip-gram model
    hs=0,
    negative=10,  # for negative sampling
    workers=2,  # no.of cores
    seed=34)

model_w2v.train(
    tokenized,
    total_examples=len(data_tidy),
    epochs=20
)


def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


wordvec_arrays = np.zeros((len(tokenized), 5))

for i in range(len(tokenized)):
    wordvec_arrays[i, :] = word_vector(tokenized[i], 5)

features = pd.DataFrame(wordvec_arrays)
print("Features shape", features.shape)

features = np.array(features)

# Split and normalize
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
print("Training Time:",  str(timeit.default_timer() - starttime)[:4])

# Metrics: Acc, Confusion Matrix

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

reversefactor = dict(zip(["need", "give", "useless"],
                         ["NEED", "GIVE", "USELESS"]))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(y_test, y_pred, rownames=[
      'Actual Classification'], colnames=['Predicted CLASSIFICATION']))
print()


# Sample Predictions

for i in [random.randint(0, len(X_test_og)) for i in range(5)]:
    print(y_pred[i])
    ind = features.tolist().index(X_test_og[i].tolist())
    print(data[ind].strip())
    print()


# Save model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
