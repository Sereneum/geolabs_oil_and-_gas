import joblib
import xgboost as xgb
import pandas as pd
import random

# Загрузка стандартизатор
scaler = joblib.load("scaler.pkl")

# Загрузка модели
model = xgb.XGBRegressor()
model.load_model("reflection_model_xgb.json")

# Подготовка новых данных (пример)
new_data = pd.DataFrame({
    "time_stuck": [random.uniform(150, 200)],
    "temperature": [random.uniform(300, 310)],
    "Ux_in": [random.uniform(-5, -4)],
    "Uy_in": [random.uniform(-3, -2)],
    "Uz_in": [random.uniform(-2, -1)],
})

# Стандартизация новых данных
new_data_scaled = scaler.transform(new_data)

# Прогнозирование
predictions = model.predict(new_data_scaled)

# Вывод прогнозов
print(new_data)
print('прогноз: ')
print(predictions)
