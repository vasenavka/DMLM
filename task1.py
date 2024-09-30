from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import pandas as pd
import mlflow
import mlflow.sklearn

# Завантаження даних
categories = ['rec.autos', 'sci.electronics', 'comp.graphics', 'rec.sport.hockey']  # вибрані категорії для прикладу
data = fetch_20newsgroups(subset='all', categories=categories)

# Виведення інформації про дані
print(f"Кількість текстів: {len(data.data)}")
print(f"Кількість категорій: {len(data.target_names)}")

# Розподіл на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Ініціалізація TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)

# Перетворення тренувального і тестового наборів
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Параметри LightGBM
params = {
    'objective': 'multiclass',
    'num_class': len(categories),  # Кількість категорій
    'metric': 'multi_logloss',  # Метріка для багатокласової класифікації
    'force_row_wise': True,  # Встановлюємо явне використання багатопоточності по рядках
    'min_gain_to_split': 0,  # Зменшення порогу мінімального приросту
    'min_data_in_leaf': 20,  # Мінімальна кількість даних у листі
    'verbosity': -1  # Вимикаємо непотрібні попередження
}

# Створення датасетів для LightGBM
train_data_lgb = lgb.Dataset(X_train_tfidf, label=y_train)
test_data_lgb = lgb.Dataset(X_test_tfidf, label=y_test)

# Навчання моделі LightGBM без параметра verbose_eval
model = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=100,
    valid_sets=[test_data_lgb],
    valid_names=['validation']  # Без verbose_eval
)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test_tfidf)
y_pred_max = [list(pred).index(max(pred)) for pred in y_pred]

# Створення DataFrame для збереження результатів
predictions_df = pd.DataFrame({'Predicted': y_pred_max, 'Actual': y_test})

# Логування моделі та результатів у MLflow
with mlflow.start_run():
    # Логування моделі через MLflow
    mlflow.sklearn.log_model(model, "lgbm_model")

    # Логування метрик
    accuracy = (predictions_df['Predicted'] == predictions_df['Actual']).mean()

    # Логування метрик в MLflow
    mlflow.log_metrics({"Accuracy": accuracy})


input_example = X_test_tfidf[0].toarray()  # Приклад вхідних даних
mlflow.sklearn.log_model(model, "lgbm_model", input_example=input_example)
