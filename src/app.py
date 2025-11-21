"""
=================================================================================
MODELO XGBOOST  - PREDICCIÓN DE PORCENTAJE DE GRASA CORPORAL
=================================================================================
Objetivo: Predecir Fat_Percentage y descubrir los factores más importantes
Autor: Danny Palacios
Link Datos: https://www.kaggle.com/datasets/jockeroika/life-style-data/data
Dataset: 20,000 filas x 54 columnas
=================================================================================
"""


# =============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import plot_importance
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print("MODELO XGBOOST - PREDICCIÓN DE FAT PERCENTAGE")
print("=" * 80)

# =============================================================================
# 2. CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================

# Cargar el dataset 
df = pd.read_csv('/workspaces/APP-web-ML--Flask-DAN/data/raw/Final_data.csv')

print("\n📊 INFORMACIÓN GENERAL DEL DATASET")
print("-" * 80)
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nPrimeras filas:")
print(df.head())

print(f"\nTipos de datos:")
print(df.dtypes.value_counts())

print(f"\nInformación detallada:")
df.info()

# Verificar variable target
print("\n🎯 ANÁLISIS DE VARIABLE TARGET: Fat_Percentage")
print("-" * 80)
print(df['Fat_Percentage'].describe())
print(f"\nValores nulos en target: {df['Fat_Percentage'].isnull().sum()}")

# =============================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

print("\n📈 ANÁLISIS EXPLORATORIO DE DATOS")
print("-" * 80)

# Valores nulos por columna
print("\n🔍 Valores nulos por columna:")
missing = df.isnull().sum()
missing_pct = 100 * df.isnull().sum() / len(df)
missing_table = pd.DataFrame({
    'Valores Nulos': missing,
    'Porcentaje': missing_pct
}).sort_values('Porcentaje', ascending=False)
print(missing_table[missing_table['Valores Nulos'] > 0])

# Estadísticas descriptivas de variables numéricas
print("\n📊 Estadísticas descriptivas:")
print(df.describe().T)

# Visualización de la distribución del target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
axes[0].hist(df['Fat_Percentage'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Fat Percentage', fontsize=12)
axes[0].set_ylabel('Frecuencia', fontsize=12)
axes[0].set_title('Distribución de Fat Percentage', fontsize=14, fontweight='bold')
axes[0].axvline(df['Fat_Percentage'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {df["Fat_Percentage"].mean():.2f}%')
axes[0].legend()

# Boxplot
axes[1].boxplot(df['Fat_Percentage'].dropna(), vert=True)
axes[1].set_ylabel('Fat Percentage', fontsize=12)
axes[1].set_title('Boxplot de Fat Percentage', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Gráfico guardado: 01_target_distribution.png")

# =============================================================================
# 4. PREPROCESAMIENTO DE DATOS
# =============================================================================

print("\n🔧 PREPROCESAMIENTO DE DATOS")
print("-" * 80)

# Crear copia para trabajar
df_clean = df.copy()

# 4.1 Manejar valores nulos en el target
print(f"\nFilas antes de eliminar nulos en target: {len(df_clean)}")
df_clean = df_clean.dropna(subset=['Fat_Percentage'])
print(f"Filas después de eliminar nulos en target: {len(df_clean)}")

# 4.2 Identificar tipos de columnas
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()

# Remover el target de las features
if 'Fat_Percentage' in numeric_cols:
    numeric_cols.remove('Fat_Percentage')

print(f"\n📋 Variables numéricas: {len(numeric_cols)}")
print(numeric_cols[:10], "...")

print(f"\n📋 Variables categóricas: {len(categorical_cols)}")
print(categorical_cols)

# 4.3 Imputación de valores nulos en features numéricas
print("\n🔢 Imputando valores nulos en variables numéricas (con la mediana)...")
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_value = df_clean[col].median()
        df_clean[col].fillna(median_value, inplace=True)
        print(f"  - {col}: {df_clean[col].isnull().sum()} nulos → completados con {median_value:.2f}")

# 4.4 Imputación de valores nulos en features categóricas
print("\n📝 Imputando valores nulos en variables categóricas (con la moda)...")
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_value = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_value, inplace=True)
        print(f"  - {col}: completado con '{mode_value}'")

# 4.5 Encoding de variables categóricas
print("\n🔤 Aplicando Label Encoding a variables categóricas...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)} categorías únicas")

# 4.6 Feature Engineering
print("\n⚙️ FEATURE ENGINEERING")
print("-" * 80)

# Calcular BMI si no existe (validación)
if 'BMI' not in df_clean.columns and 'Weight (kg)' in df_clean.columns and 'Height (m)' in df_clean.columns:
    df_clean['BMI'] = df_clean['Weight (kg)'] / (df_clean['Height (m)'] ** 2)
    print("✅ BMI calculado")

# Crear nuevas features relevantes
if 'Calories_Burned' in df_clean.columns and 'Session_Duration (hours)' in df_clean.columns:
    df_clean['Calories_Per_Hour'] = df_clean['Calories_Burned'] / (df_clean['Session_Duration (hours)'] + 0.001)
    print("✅ Calories_Per_Hour creado")

if 'Avg_BPM' in df_clean.columns and 'Resting_BPM' in df_clean.columns:
    df_clean['BPM_Elevation'] = df_clean['Avg_BPM'] - df_clean['Resting_BPM']
    print("✅ BPM_Elevation creado")

if 'Proteins' in df_clean.columns and 'Weight (kg)' in df_clean.columns:
    df_clean['Protein_Per_Kg'] = df_clean['Proteins'] / (df_clean['Weight (kg)'] + 0.001)
    print("✅ Protein_Per_Kg creado")

if 'Calories' in df_clean.columns and 'Weight (kg)' in df_clean.columns:
    df_clean['Calorie_Density'] = df_clean['Calories'] / (df_clean['Weight (kg)'] + 0.001)
    print("✅ Calorie_Density creado")

# Ratio de macronutrientes
if all(col in df_clean.columns for col in ['Carbs', 'Proteins', 'Fats']):
    total_macros = df_clean['Carbs'] + df_clean['Proteins'] + df_clean['Fats'] + 0.001
    df_clean['Carbs_Ratio'] = df_clean['Carbs'] / total_macros
    df_clean['Protein_Ratio'] = df_clean['Proteins'] / total_macros
    df_clean['Fat_Ratio'] = df_clean['Fats'] / total_macros
    print("✅ Ratios de macronutrientes creados")

print(f"\n📊 Dimensiones finales del dataset: {df_clean.shape}")

# =============================================================================
# 5. PREPARACIÓN DE DATOS PARA MODELADO
# =============================================================================

print("\n🎲 PREPARACIÓN DE DATOS PARA MODELADO")
print("-" * 80)

# Separar features y target
X = df_clean.drop('Fat_Percentage', axis=1)
y = df_clean['Fat_Percentage']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n✂️ División de datos:")
print(f"  - Entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

# =============================================================================
# 6. ANÁLISIS DE CORRELACIONES
# =============================================================================

print("\n🔗 ANÁLISIS DE CORRELACIONES CON FAT_PERCENTAGE")
print("-" * 80)

# Calcular correlaciones con el target
correlations = df_clean.corr()['Fat_Percentage'].sort_values(ascending=False)
print("\n🔝 Top 15 variables más correlacionadas:")
print(correlations.head(15))

print("\n🔻 Top 10 variables menos correlacionadas:")
print(correlations.tail(10))

# Visualizar top correlaciones
fig, ax = plt.subplots(figsize=(10, 8))
top_corr = correlations.head(20)
colors = ['green' if x > 0 else 'red' for x in top_corr.values]
top_corr.plot(kind='barh', color=colors, ax=ax)
ax.set_xlabel('Correlación con Fat_Percentage', fontsize=12)
ax.set_title('Top 20 Variables más Correlacionadas con Fat Percentage', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('02_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Gráfico guardado: 02_correlations.png")

# =============================================================================
# 7. ENTRENAMIENTO DEL MODELO BASE XGBOOST
# =============================================================================

print("\n🚀 ENTRENAMIENTO DEL MODELO BASE XGBOOST")
print("-" * 80)

# Configuración inicial del modelo
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Función objetivo para regresión
    n_estimators=100,              # Número de árboles
    learning_rate=0.1,             # Tasa de aprendizaje
    max_depth=6,                   # Profundidad máxima de árboles
    min_child_weight=1,            # Peso mínimo en nodos hijos
    subsample=0.8,                 # Fracción de muestras por árbol
    colsample_bytree=0.8,          # Fracción de features por árbol
    random_state=42,
    n_jobs=-1,                     # Usar todos los cores disponibles
    verbosity=1
)

print("⏳ Entrenando modelo base...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Predicciones
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Evaluación del modelo base
print("\n📊 EVALUACIÓN DEL MODELO BASE")
print("-" * 80)

# Métricas de entrenamiento
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

print("🏋️ CONJUNTO DE ENTRENAMIENTO:")
print(f"  - RMSE: {rmse_train:.4f}")
print(f"  - MAE:  {mae_train:.4f}")
print(f"  - R²:   {r2_train:.4f}")
print(f"  - MAPE: {mape_train:.2f}%")

# Métricas de prueba
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("\n🧪 CONJUNTO DE PRUEBA:")
print(f"  - RMSE: {rmse_test:.4f}")
print(f"  - MAE:  {mae_test:.4f}")
print(f"  - R²:   {r2_test:.4f}")
print(f"  - MAPE: {mape_test:.2f}%")

# Verificar overfitting
overfit_diff = abs(r2_train - r2_test)
print(f"\n⚠️ Diferencia R² (Train - Test): {overfit_diff:.4f}")
if overfit_diff > 0.1:
    print("   → Posible overfitting. Considera ajustar hiperparámetros.")
else:
    print("   → ✅ Modelo generaliza bien.")

# =============================================================================
# 8. VALIDACIÓN CRUZADA
# =============================================================================

print("\n🔄 VALIDACIÓN CRUZADA (K-Fold = 5)")
print("-" * 80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluar con diferentes métricas
cv_rmse = -cross_val_score(xgb_model, X_train, y_train, 
                           cv=kfold, scoring='neg_root_mean_squared_error')
cv_mae = -cross_val_score(xgb_model, X_train, y_train, 
                          cv=kfold, scoring='neg_mean_absolute_error')
cv_r2 = cross_val_score(xgb_model, X_train, y_train, 
                        cv=kfold, scoring='r2')

print(f"📊 Resultados de Cross-Validation:")
print(f"  - RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
print(f"  - MAE:  {cv_mae.mean():.4f} (+/- {cv_mae.std():.4f})")
print(f"  - R²:   {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

# =============================================================================
# 9. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================

print("\n🎛️ OPTIMIZACIÓN DE HIPERPARÁMETROS (Grid Search)")
print("-" * 80)
print("⏳ Este proceso puede tomar varios minutos...")

from sklearn.model_selection import GridSearchCV

# Definir grid de parámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

# Grid Search con validación cruzada
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
print("-" * 80)
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\n📈 Mejor RMSE (CV): {-grid_search.best_score_:.4f}")

# Modelo optimizado
best_model = grid_search.best_estimator_

# Predicciones con modelo optimizado
y_pred_train_opt = best_model.predict(X_train)
y_pred_test_opt = best_model.predict(X_test)

# Evaluación del modelo optimizado
print("\n📊 EVALUACIÓN DEL MODELO OPTIMIZADO")
print("-" * 80)

rmse_test_opt = np.sqrt(mean_squared_error(y_test, y_pred_test_opt))
mae_test_opt = mean_absolute_error(y_test, y_pred_test_opt)
r2_test_opt = r2_score(y_test, y_pred_test_opt)
mape_test_opt = np.mean(np.abs((y_test - y_pred_test_opt) / y_test)) * 100

print("🧪 CONJUNTO DE PRUEBA (Modelo Optimizado):")
print(f"  - RMSE: {rmse_test_opt:.4f} (Base: {rmse_test:.4f})")
print(f"  - MAE:  {mae_test_opt:.4f} (Base: {mae_test:.4f})")
print(f"  - R²:   {r2_test_opt:.4f} (Base: {r2_test:.4f})")
print(f"  - MAPE: {mape_test_opt:.2f}% (Base: {mape_test:.2f}%)")

improvement = ((rmse_test - rmse_test_opt) / rmse_test) * 100
print(f"\n✨ Mejora en RMSE: {improvement:.2f}%")

# =============================================================================
# 10. IMPORTANCIA DE FEATURES
# =============================================================================

print("\n⭐ IMPORTANCIA DE FEATURES")
print("-" * 80)

# Obtener importancia de features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔝 Top 20 Features Más Importantes:")
print(feature_importance.head(20).to_string(index=False))

# Visualización de importancia
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Gráfico 1: Top 20 features
top_20 = feature_importance.head(20)
axes[0].barh(range(len(top_20)), top_20['Importance'], color='skyblue')
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20['Feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Importancia', fontsize=12)
axes[0].set_title('Top 20 Features Más Importantes', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Gráfico 2: XGBoost built-in importance plot
plot_importance(best_model, max_num_features=20, ax=axes[1], 
                importance_type='gain', show_values=False)
axes[1].set_title('Feature Importance (Gain)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('03_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Gráfico guardado: 03_feature_importance.png")

# =============================================================================
# 11. ANÁLISIS SHAP (Interpretabilidad)
# =============================================================================

print("\n🔍 ANÁLISIS SHAP - INTERPRETABILIDAD DEL MODELO")
print("-" * 80)
print("⏳ Calculando valores SHAP (puede tomar un momento)...")

# Crear explainer SHAP
explainer = shap.TreeExplainer(best_model)

# Calcular SHAP values (usar muestra para eficiencia)
sample_size = min(1000, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Summary plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('04_shap_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Gráfico guardado: 04_shap_importance.png")

# SHAP summary plot detallado
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('SHAP Summary Plot - Impacto y Distribución', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('05_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Gráfico guardado: 05_shap_summary.png")

# =============================================================================
# 12. VISUALIZACIÓN DE PREDICCIONES
# =============================================================================

print("\n📉 VISUALIZACIÓN DE PREDICCIONES")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Predicciones vs Valores Reales
axes[0, 0].scatter(y_test, y_pred_test_opt, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Predicción Perfecta')
axes[0, 0].set_xlabel('Fat Percentage Real', fontsize=11)
axes[0, 0].set_ylabel('Fat Percentage Predicho', fontsize=11)
axes[0, 0].set_title(f'Predicciones vs Valores Reales\n(R² = {r2_test_opt:.4f})', 
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Distribución de residuos
residuals = y_test - y_pred_test_opt
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
axes[0, 1].set_xlabel('Residuos (Real - Predicho)', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].set_title(f'Distribución de Residuos\n(Media = {residuals.mean():.4f})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Residuos vs Predicciones
axes[1, 0].scatter(y_pred_test_opt, residuals, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Fat Percentage Predicho', fontsize=11)
axes[1, 0].set_ylabel('Residuos', fontsize=11)
axes[1, 0].set_title('Residuos vs Predicciones\n(Homocedasticidad)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Errores absolutos
errors = np.abs(residuals)
axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(x=mae_test_opt, color='red', linestyle='--', 
                   linewidth=2, label=f'MAE = {mae_test_opt:.4f}')
axes[1, 1].set_xlabel('Error Absoluto', fontsize=11)
axes[1, 1].set_ylabel('Frecuencia', fontsize=11)
axes[1, 1].set_title('Distribución de Errores Absolutos', 
                     fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_predictions_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Gráfico guardado: 06_predictions_analysis.png")

# =============================================================================
# 13. GUARDAR EL MODELO
# =============================================================================

print("\n💾 GUARDANDO MODELO Y RESULTADOS")
print("-" * 80)

# Guardar modelo
import joblib
joblib.dump(best_model, 'xgboost_fat_percentage_model.pkl')
print("✅ Modelo guardado: xgboost_fat_percentage_model.pkl")

# Guardar encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Encoders guardados: label_encoders.pkl")

# Guardar feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance guardado: feature_importance.csv")

# Guardar métricas
metrics_summary = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
    'Train': [
        np.sqrt(mean_squared_error(y_train, y_pred_train_opt)),
        mean_absolute_error(y_train, y_pred_train_opt),
        r2_score(y_train, y_pred_train_opt),
        np.mean(np.abs((y_train - y_pred_train_opt) / y_train)) * 100
    ],
    'Test': [rmse_test_opt, mae_test_opt, r2_test_opt, mape_test_opt]
})
metrics_summary.to_csv('model_metrics.csv', index=False)
print("✅ Métricas guardadas: model_metrics.csv")

# =============================================================================
# 14. RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 80)
print("📋 RESUMEN FINAL DEL MODELO")
print("=" * 80)

print(f"""
🎯 OBJETIVO: Predecir Fat_Percentage corporal

📊 DATOS:
  - Total de muestras: {len(df_clean):,}
  - Features utilizados: {X.shape[1]}
  - Train/Test split: 80%/20%

🏆 MEJOR MODELO (XGBoost Optimizado):
  - n_estimators: {best_model.n_estimators}
  - max_depth: {best_model.max_depth}
  - learning_rate: {best_model.learning_rate}

📈 MÉTRICAS DE EVALUACIÓN (Test Set):
  - RMSE: {rmse_test_opt:.4f}
  - MAE:  {mae_test_opt:.4f}
  - R²:   {r2_test_opt:.4f}
  - MAPE: {mape_test_opt:.2f}%

⭐ TOP 5 FACTORES MÁS IMPORTANTES:
""")

for i, row in feature_importance.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"""
💡 INTERPRETACIÓN:
  - El modelo explica el {r2_test_opt*100:.2f}% de la varianza en Fat_Percentage
  - Error promedio de predicción: {mae_test_opt:.2f} puntos porcentuales
  - {'✅ Buen ajuste' if r2_test_opt > 0.7 else '⚠️ Considerar más features o datos'}

📁 ARCHIVOS GENERADOS:
  - xgboost_fat_percentage_model.pkl (modelo entrenado)
  - label_encoders.pkl (encoders para variables categóricas)
  - feature_importance.csv (importancia de variables)
  - model_metrics.csv (métricas de evaluación)
  - 6 gráficos PNG (análisis visual completo)

🚀 PARA USAR EL MODELO:
""")

print("""
# Cargar el modelo
import joblib
model = joblib.load('xgboost_fat_percentage_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Hacer predicciones
new_data = pd.DataFrame({...})  # Tus nuevos datos
prediction = model.predict(new_data)
print(f'Fat Percentage predicho: {prediction[0]:.2f}%')
""")

print("=" * 80)
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 80)

# =============================================================================
# 15. ANÁLISIS ADICIONAL: INSIGHTS SOBRE FAT PERCENTAGE
# =============================================================================

print("\n\n🔬 ANÁLISIS ADICIONAL: INSIGHTS CLAVE SOBRE FAT PERCENTAGE")
print("=" * 80)

# Análisis por rangos de Fat Percentage
df_clean['Fat_Category'] = pd.cut(
    df_clean['Fat_Percentage'],
    bins=[0, 15, 25, 35, 100],
    labels=['Bajo (<15%)', 'Normal (15-25%)', 'Alto (25-35%)', 'Muy Alto (>35%)']
)

print("\n📊 Distribución por categoría:")
print(df_clean['Fat_Category'].value_counts().sort_index())

# Análisis de las top features por categoría
if 'BMI' in df_clean.columns:
    print("\n🏋️ Promedio de BMI por categoría de grasa:")
    print(df_clean.groupby('Fat_Category')['BMI'].mean().sort_index())

if 'Workout_Frequency (days/week)' in df_clean.columns:
    print("\n🏃 Promedio de frecuencia de ejercicio por categoría:")
    print(df_clean.groupby('Fat_Category')['Workout_Frequency (days/week)'].mean().sort_index())

if 'Protein_Ratio' in df_clean.columns:
    print("\n🥩 Promedio de ratio de proteínas por categoría:")
    print(df_clean.groupby('Fat_Category')['Protein_Ratio'].mean().sort_index())

# =============================================================================
# 16. RECOMENDACIONES Y PRÓXIMOS PASOS
# =============================================================================

print("\n\n💡 RECOMENDACIONES Y PRÓXIMOS PASOS")
print("=" * 80)

recommendations = """
📌 FACTORES CLAVE IDENTIFICADOS:
   Los factores más importantes para predecir Fat_Percentage incluyen
   variables relacionadas con:
   - Composición corporal (BMI, Weight, Height)
   - Nutrición (ratios de macronutrientes, ingesta calórica)
   - Actividad física (frecuencia, intensidad, tipo de ejercicio)
   - Metabolismo (BPM en reposo, edad)

🎯 PARA MEJORAR AÚN MÁS EL MODELO:
   1. Recolectar más datos si R² < 0.80
   2. Crear interacciones entre features importantes (ej: BMI * Workout_Frequency)
   3. Probar ensemble con otros algoritmos (Random Forest, LightGBM)
   4. Aplicar feature selection más riguroso (eliminar features poco importantes)
   5. Considerar técnicas de regularización adicionales

📊 APLICACIONES PRÁCTICAS:
   - Sistemas de recomendación personalizados de dieta y ejercicio
   - Evaluación de progreso en programas fitness
   - Identificación de factores de riesgo para obesidad
   - Optimización de planes nutricionales

⚠️ CONSIDERACIONES IMPORTANTES:
   - Este modelo es para propósitos educativos/informativos
   - No reemplaza evaluación médica profesional
   - Los resultados dependen de la calidad de los datos de entrada
   - Validar en datos nuevos antes de uso en producción
"""

print(recommendations)

# =============================================================================
# 17. FUNCIÓN DE PREDICCIÓN EJEMPLO
# =============================================================================

print("\n\n🔧 FUNCIÓN DE PREDICCIÓN PERSONALIZADA")
print("=" * 80)

example_code = """
def predict_fat_percentage(model, encoders, user_data):
    \"\"\"
    Predice el porcentaje de grasa corporal para un nuevo usuario
    
    Parameters:
    -----------
    model : XGBoost model
        Modelo entrenado
    encoders : dict
        Diccionario de LabelEncoders
    user_data : dict
        Datos del usuario con todas las features
    
    Returns:
    --------
    float : Porcentaje de grasa predicho
    dict : Información adicional (categoría, recomendaciones)
    \"\"\"
    import pandas as pd
    import numpy as np
    
    # Convertir a DataFrame
    df = pd.DataFrame([user_data])
    
    # Aplicar encoders a variables categóricas
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))
    
    # Crear features engineered (igual que en entrenamiento)
    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    
    if 'Calories_Per_Hour' not in df.columns:
        df['Calories_Per_Hour'] = df['Calories_Burned'] / (df['Session_Duration (hours)'] + 0.001)
    
    # ... (agregar todas las features creadas en el entrenamiento)
    
    # Predecir
    prediction = model.predict(df)[0]
    
    # Categorizar
    if prediction < 15:
        category = "Bajo"
        recommendation = "Mantén tu rutina actual"
    elif prediction < 25:
        category = "Normal"
        recommendation = "Excelente rango saludable"
    elif prediction < 35:
        category = "Alto"
        recommendation = "Considera aumentar actividad física"
    else:
        category = "Muy Alto"
        recommendation = "Consulta con un profesional de salud"
    
    return {
        'fat_percentage': round(prediction, 2),
        'category': category,
        'recommendation': recommendation
    }

# EJEMPLO DE USO:
# ================

# Cargar modelo y encoders
import joblib
model = joblib.load('xgboost_fat_percentage_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Datos de ejemplo de un nuevo usuario
new_user = {
    'Age': 30,
    'Gender': 'Male',
    'Weight (kg)': 80,
    'Height (m)': 1.75,
    'Workout_Frequency (days/week)': 4,
    'Calories': 2500,
    'Proteins': 150,
    'Carbs': 250,
    'Fats': 80,
    # ... (todas las demás features requeridas)
}

# Hacer predicción
result = predict_fat_percentage(model, encoders, new_user)
print(f"Porcentaje de grasa predicho: {result['fat_percentage']}%")
print(f"Categoría: {result['category']}")
print(f"Recomendación: {result['recommendation']}")
"""

print(example_code)

print("\n" + "=" * 80)
print("🎉 SCRIPT COMPLETADO - ¡Listo para usar!")
print("=" * 80)

print("""
📚 DOCUMENTACIÓN ADICIONAL:
   - XGBoost: https://xgboost.readthedocs.io/
   - SHAP: https://shap.readthedocs.io/
   - Scikit-learn: https://scikit-learn.org/

💬 ¿Preguntas o ajustes? Puedes:
   - Modificar hiperparámetros en la sección 9
   - Ajustar feature engineering en la sección 4.6
   - Cambiar la estrategia de validación en la sección 8
   - Personalizar visualizaciones en las secciones 10-12
""")
