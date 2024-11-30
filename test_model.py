
from joblib import load

emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

loaded_model = load('naive_bayes_emotion_model.pkl')
loaded_vectorizer = load('tfidf_vectorizer.pkl')

new_texts = ["i feel a bit intimidated and out of my league due to his experience", "This is so sad and heartbreaking."]
new_texts_vectorized = loaded_vectorizer.transform(new_texts)

predictions = loaded_model.predict(new_texts_vectorized)

predicted_emotions = [emotion_map[label] for label in predictions]

print("Predicted emotions:", predicted_emotions)
