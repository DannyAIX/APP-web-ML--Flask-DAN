# =============================================================================
# PROYECTO: XGBoost - Optimización rápida con RandomizedSearchCV
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
# Reemplaza con la ruta de tu dataset
df = pd.read_csv('/workspaces/APP-web-ML--Flask-DAN/data/raw/Final_data.csv')

# Separa características y target
X = df.drop(columns=['Fat_Percentage'])
y = df['Fat_Percentage']

# =============================================================================
# 2. PREPROCESAMIENTO (convertir columnas categóricas)
# =============================================================================
# Identificar columnas categóricas
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].astype('category')

# División train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# 3. DEFINIR MODELO XGBOOST
# =============================================================================
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42,
    tree_method='hist',          # Cambiar a 'gpu_hist' si tienes GPU
    enable_categorical=True,     # Para columnas categóricas
    early_stopping_rounds=20     # ⚡ Early stopping
)

# =============================================================================
# 4. PARAMETROS PARA RANDOMIZED SEARCH
# =============================================================================
param_distributions = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# =============================================================================
# 5. RANDOMIZED SEARCH
# =============================================================================
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=10,  # menos combinaciones
    scoring='neg_root_mean_squared_error',
    cv=2,       # menos folds
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Entrenamiento con RandomizedSearchCV
random_search.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# =============================================================================
# 6. MEJOR MODELO
# =============================================================================
best_model = random_search.best_estimator_
print("✅ Mejor RMSE en validación (CV):", -random_search.best_score_)

# =============================================================================
# 7. PREDICCIONES Y EVALUACIÓN
# =============================================================================
y_pred_test = best_model.predict(X_test)


from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"📊 Evaluación en Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")

# =============================================================================
# 8. IMPORTANCIA DE VARIABLES
# =============================================================================
xgb.plot_importance(best_model, max_num_features=10, importance_type='weight')
plt.title('Top 10 Variables más importantes')
plt.show()

# =============================================================================
# 9. GUARDAR MODELO
# =============================================================================
joblib.dump(best_model, 'xgb_best_model.pkl')
print("💾 Modelo guardado como 'xgb_best_model.pkl'")

# Para cargar el modelo más adelante:
# best_model_loaded = joblib.load('xgb_best_model.pkl')