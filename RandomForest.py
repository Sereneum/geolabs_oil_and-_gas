import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("Ar_300_rough_collision.dat", delim_whitespace=True)
data = pd.concat([pd.DataFrame([data.columns.values], columns=data.columns), data], ignore_index=True)
data.columns = ['An', 'T','K','Ux_in','Uy_in','Uz_in','Ux_out','Uy_out','Uz_out']

for col in data.columns:
    data[col] = data[col].astype(float)
data.dropna(inplace=True)

X = data[["T","K","Ux_in", "Uy_in", "Uz_in"]]
y = data[["Ux_out", "Uy_out", "Uz_out"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
rf_random.best_params_

model_rf = RandomForestRegressor(n_estimators= 800,min_samples_split=2,min_samples_leaf= 4,max_features='sqrt',max_depth=10,bootstrap=True, random_state=42)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Regressor - R^2: {r2_rf:.4f}, MSE: {mse_rf:.4f}")