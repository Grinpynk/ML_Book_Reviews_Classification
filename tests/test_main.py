
import pytest
import pandas as pd
from ml_book_reviews_classification.main import pd, TfidfVectorizer, LogisticRegression

def test_data_loading():
    data = pd.read_csv('ml_book_reviews_classification/data/reviews.csv')
    assert not data.empty, "Датасет не должен быть пустым"
    assert 'review' in data.columns, "Датасет должен содержать колонку 'review'"

def test_model_prediction():
    reviews = ["Отличная книга!", "Не понравилось"]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(reviews)
    model = LogisticRegression()
    y = [1, 0]
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 2, "Должны быть предсказания для двух отзывов"
