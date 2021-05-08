import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mdl = None
with open("model.pkl","rb") as f:
    mdl = pickle.load(f)

vectorizer = CountVectorizer(analyzer="word",lowercase=False)
scaler = StandardScaler()

tweet = input("Enter input")
vectorized = vectorizer.fit_transform([tweet]).toarray()
print("vecorized shape ", vectorized.shape)
#_,X_test, _, _ = train_test_split(vectorized, vectorized, train_size=0.5)
scaled = scaler.fit_transform(vectorized)

prediction = mdl.predict(scaled)
print(prediction)
