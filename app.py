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


# Mapeos para convertir valores del formulario (si envías números)
FORM_MAPPINGS = {
'gender': {'1': 'Male', '2': 'Female', 'Male': 'Male', 'Female': 'Female'},
'workout_type': {'1': 'Cardio', '2': 'Strength', '3': 'HIIT', '4': 'Yoga',
'Cardio': 'Cardio', 'Strength': 'Strength', 'HIIT': 'HIIT', 'Yoga': 'Yoga'},
'experience_level': {'1': '1', '2': '2', '3': '3', 'Beginner': '1', 'Intermediate': '2', 'Advanced': '3'}
}


NUMERIC_FIELDS = {
'age': float,
'weight': float,
'height': float,
'workout_frequency': float,
'calories': float,
'proteins': float,
'carbs': float,
'fats': float,
'resting_bpm': float,
'avg_bpm': float,
'session_duration': float,
'calories_burned': float
}




def safe_apply_label_encoder(col_series, encoder):
"""
Transforma una Series con un LabelEncoder de forma segura: si aparece un valor
desconocido se reemplaza por la clase más frecuente (primer elemento de classes_).
"""
col_series = col_series.astype(str)
valid = set(encoder.classes_.tolist())
default = encoder.classes_[0]
col_series = col_series.apply(lambda x: x if x in valid else default)
app.run(host='0.0.0.0', port=port, debug=False)