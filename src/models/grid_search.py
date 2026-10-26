import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 1. Read the scaled training inputs and the training target
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

# 2. The model and the parameter values we want to try
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

# 3. Try every combination with 5-fold cross-validation
grid = GridSearchCV(model, param_grid, cv=5, scoring="r2")
grid.fit(X_train, y_train)

# 4. Save the best parameters in the models folder
joblib.dump(grid.best_params_, "models/best_params.pkl")

print("Best parameters:", grid.best_params_)
print("Best cross-validated R2:", round(grid.best_score_, 3))
