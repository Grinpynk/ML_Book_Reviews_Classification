
# PEP 8 форматированный код для ML_Book_Reviews_Classification
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Загружаем набор данных
DATA_PATH = 'data/reviews.csv'  # Путь к набору данных
data = pd.read_csv(DATA_PATH)

# Предобрабатываем данные
data.dropna(subset=['review'], inplace=True)  # Удаляем строки с отсутствующими отзывами
X = data['review']  # Тексты отзывов
y = data['sentiment']  # Метки классов

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем текстовые данные в числовой формат (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Обучаем модель логистической регрессии
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Делаем прогнозы
y_pred = model.predict(X_test_tfidf)

# Выводим отчет о качестве модели
print(classification_report(y_test, y_pred))
