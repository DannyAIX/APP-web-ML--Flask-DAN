from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Inicializar Flask
app = Flask(__name__)

# Configuración
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave-secreta')

# ============================
# CARGA DE MODELO Y ENCODERS
# ============================
try:
    model = joblib.load('models/xgboost_fat_percentage_model.pkl')
    encoders = joblib.load('models/label_encoders.pkl')
    print("✅ Modelo y encoders cargados exitosamente")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    model = None
    encoders = {}

# ============================
# FUNCIONES AUXILIARES
# ============================

def preprocess_input(data):
    """
    Normaliza claves, crea DataFrame y aplica transformaciones.
    """
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

    safe = {}
    for key in expected_map:
        safe_key = expected_map[key]
        safe[safe_key] = data.get(key, data.get(safe_key, None))

    df = pd.DataFrame([safe])

    # Aplicar encoders a columnas categóricas
    for col, enc in encoders.items():
        if col in df:
            df[col] = enc.transform(df[col].astype(str))

    return df

# ============================
# RUTAS
# ============================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    data = request.form.to_dict()
    df = preprocess_input(data)

    # Predicción
    pred = model.predict(df)[0]
    pred = round(float(pred), 2)

    # Determinar categoría
    if pred < 10:
        category = "Muy Bajo"
        color = "primary"
        icon = "⚪"
        recommendation = "Tu nivel de grasa es muy bajo. Prioriza buena nutrición."
    elif pred < 20:
        category = "Atleta"
        color = "success"
        icon = "🏅"
        recommendation = "Excelente composición corporal. Sigue así."
    elif pred < 25:
        category = "Fitness"
        color = "info"
        icon = "💪"
        recommendation = "Buen nivel, ideal para la mayoría de personas."
    elif pred < 31:
        category = "Normal"
        color = "warning"
        icon = "🙂"
        recommendation = "Estás dentro del rango promedio saludable."
    else:
        category = "Alto"
        color = "danger"
        icon = "⚠️"
        recommendation = "Reduce grasa corporal con ejercicio y nutrición adecuada."

    result = {
        "fat_percentage": pred,
        "category": category,
        "color": color,
        "icon": icon,
        "recommendation": recommendation
    }

    return render_template("result.html", result=result)

# ============================
# EJECUCIÓN
# ============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)