from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ==========================
# CARGA DE MODELO
# ==========================
try:
    model = joblib.load('models/xgb_best_model.pkl')
    print("✅ Modelo cargado")
except Exception as e:
    print("❌ Error cargando modelo:", e)
    model = None

# ==========================
# MAPEOS
# ==========================
FORM_MAPPINGS = {
    'gender': {
        '1': 'Male', '2': 'Female',
        'Male': 'Male', 'Female': 'Female'
    },
    'workout_type': {
        '1': 'Cardio', '2': 'Strength', '3': 'HIIT', '4': 'Yoga',
        'Cardio': 'Cardio', 'Strength': 'Strength', 'HIIT': 'HIIT', 'Yoga': 'Yoga'
    },
    'experience_level': {
        '1': '1', '2': '2', '3': '3',
        'Beginner': '1', 'Intermediate': '2', 'Advanced': '3'
    }
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

# ==========================
# RUTA PRINCIPAL (/)
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

# ==========================
# RUTA DE PREDICCIÓN (/predict)
# ==========================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    data = request.get_json()

    # Conversión numérica
    for key, cast in NUMERIC_FIELDS.items():
        if key in data:
            try:
                data[key] = cast(data[key])
            except:
                data[key] = 0

    # Mapeos de categorías
    for key, mapping in FORM_MAPPINGS.items():
        if key in data:
            data[key] = mapping.get(data[key], list(mapping.values())[0])

    # Convertir a DataFrame
    df = pd.DataFrame([data])

    # Convertir columnas categóricas a category
    for col in FORM_MAPPINGS.keys():
        if col in df.columns:
            df[col] = df[col].astype('category')

    # ======================
    # HACER LA PREDICCIÓN
    # ======================
    try:
        pred = float(model.predict(df)[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "predicted_bodyfat": round(pred, 2)
    })

# ==========================
# RUN APP
# ==========================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)