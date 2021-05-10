import pickle

clf = pickle.load(open("./model/clf.pkl", 'rb'))
vectorizer = pickle.load(open("./model/vectorizer.pkl", 'rb'))


def infer(tweet: str):
    res = clf.predict(vectorizer.transform([tweet]))
    if res == ["useless"]:
        return "other"
    else:
        return res[0]


if __name__ == "__main__":
    print(infer(input("Enter input: ")))
