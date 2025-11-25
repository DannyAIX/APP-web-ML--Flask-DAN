from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os


# Inicializar Flask
app = Flask(__name__)


# Configuración
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave-secreta')


# Cargar modelo y encoders al iniciar la aplicación
try:
model = joblib.load('models/xgboost_fat_percentage_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')
print("✅ Modelo y encoders cargados exitosamente")
except Exception as e:
print(f"❌ Error al cargar modelo: {e}")
model = None
encoders = {}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def preprocess_input(data):
"""
Normaliza claves, crea DataFrame y aplica transformaciones.
"""
# Normalizar claves esperadas (sin espacios ni paréntesis)
expected_map = {
'Age': 'Age',
'Gender': 'Gender',
'Weight_kg': 'Weight_kg',
'Height_m': 'Height_m',
'Workout_Frequency': 'Workout_Frequency',
'Calories': 'Calories',
'Proteins': 'Proteins',
'Carbs': 'Carbs',
'Fats': 'Fats',
'Workout_Type': 'Workout_Type',
'Experience_Level': 'Experience_Level',
'Resting_BPM': 'Resting_BPM',
'Avg_BPM': 'Avg_BPM',
'Session_Duration_hours': 'Session_Duration_hours',
'Calories_Burned': 'Calories_Burned'
}


# Build df with default safe values
safe = {}
for k in expected_map:
safe_key = expected_map[k]
safe[safe_key] = data.get(k, data.get(safe_key, None))
app.run(host='0.0.0.0', port=port, debug=False)