import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

df = pd.read_csv("D:/mew/facebook_cmt/facebook_cmt/data/facebook_comments_dataset.csv")

X = df["comment"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

# Đảm bảo thư mục model tồn tại
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Lưu model vào đúng thư mục hiện tại (model/)
model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
joblib.dump(model, model_path)
