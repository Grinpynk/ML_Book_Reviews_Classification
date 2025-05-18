
# ML Book Reviews Classification

Этот проект представляет собой классификацию отзывов на книги как положительные или отрицательные.

## Основные файлы:
- `main.py` - основной код модели.
- `data/reviews.csv` - пример данных с отзывами.
- `.flake8` - конфигурация для линтера flake8.
- `.github/workflows/lint.yml` - конфигурация GitHub Actions для автоматической проверки стиля кода.

## Как запустить:
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

2. Запустите модель:
   ```bash
   python ml_book_reviews_classification/main.py
   ```

3. Для проверки стиля:
   ```bash
   flake8
   ```
