
# ML Book Reviews Classification

Этот проект представляет собой классификацию отзывов на книги как положительные или отрицательные.

## Запуск проекта

1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустите модель:
   ```bash
   python ml_book_reviews_classification/main.py
   ```

3. Запустите тесты:
   ```bash
   pytest tests/
   ```

## Структура данных

Пример данных (`reviews.csv`):
```
review,sentiment
Отличная книга,положительный
Не понравилось,отрицательный
```

## API и структура модели
- Модель использует TF-IDF для преобразования текста и LogisticRegression для классификации.
