# -*- coding: utf-8 -*-
# !pip install scikit-learn joblib xgboost pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load data
data = pd.read_csv("Ar_300_rough_collision.dat", sep=" ", header=None)
data.columns = ["atom_id", "time_stuck", "temperature", "Ux_in", "Uy_in", "Uz_in", "Ux_out", "Uy_out", "Uz_out"]

# Separate features and target
X = data[["time_stuck", "temperature", "Ux_in", "Uy_in", "Uz_in"]]
y = data[["Ux_out", "Uy_out", "Uz_out"]]

# Standardize features
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define XGBoost model
model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=10, random_state=3)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred, multioutput="variance_weighted")
mse = mean_squared_error(y_test, y_pred)
print(f"R2 = {r2:.4f}, MSE = {mse:.4f}")

# Feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# Save model and scaler
model.save_model("reflection_model_xgb_500.json")
joblib.dump(scaler, 'scaler_500.pkl')
