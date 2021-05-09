import pickle

clf = pickle.load(open("./model/clf.pkl", 'rb'))
vectorizer = pickle.load(open("./model/vectorizer.pkl", 'rb'))

tweet = input("Enter input: ")

prediction = clf.predict(vectorizer.transform([tweet]))
print(prediction)
