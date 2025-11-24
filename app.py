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
    encoders = None

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def preprocess_input(data):
    """
    Preprocesa los datos de entrada para el modelo
    """
    df = pd.DataFrame([data])
    
    # Aplicar encoders a variables categóricas
    categorical_cols = ['Gender', 'Workout_Type', 'diet_type', 'cooking_method']
    
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except:
                # Si hay un valor no visto, usar el más común
                df[col] = 0
    
    # Feature Engineering (igual que en entrenamiento)
    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    
    if 'Calories_Per_Hour' not in df.columns and 'Calories_Burned' in df.columns:
        df['Calories_Per_Hour'] = df['Calories_Burned'] / (df.get('Session_Duration (hours)', 1) + 0.001)
    
    if 'BPM_Elevation' not in df.columns:
        df['BPM_Elevation'] = df.get('Avg_BPM', 120) - df.get('Resting_BPM', 70)
    
    if 'Protein_Per_Kg' not in df.columns:
        df['Protein_Per_Kg'] = df.get('Proteins', 100) / (df['Weight (kg)'] + 0.001)
    
    if 'Calorie_Density' not in df.columns:
        df['Calorie_Density'] = df.get('Calories', 2000) / (df['Weight (kg)'] + 0.001)
    
    # Ratios de macronutrientes
    total_macros = df.get('Carbs', 200) + df.get('Proteins', 100) + df.get('Fats', 60) + 0.001
    df['Carbs_Ratio'] = df.get('Carbs', 200) / total_macros
    df['Protein_Ratio'] = df.get('Proteins', 100) / total_macros
    df['Fat_Ratio'] = df.get('Fats', 60) / total_macros
    
    return df

def categorize_fat_percentage(fat_pct):
    """
    Categoriza el porcentaje de grasa y da recomendaciones
    """
    if fat_pct < 15:
        return {
            'category': 'Bajo',
            'color': 'success',
            'recommendation': 'Tu porcentaje de grasa es bajo. Mantén tu rutina actual.',
            'icon': '💪'
        }
    elif fat_pct < 25:
        return {
            'category': 'Normal',
            'color': 'info',
            'recommendation': '¡Excelente! Estás en un rango saludable.',
            'icon': '✅'
        }
    elif fat_pct < 35:
        return {
            'category': 'Alto',
            'color': 'warning',
            'recommendation': 'Considera aumentar tu actividad física y revisar tu dieta.',
            'icon': '⚠️'
        }
    else:
        return {
            'category': 'Muy Alto',
            'color': 'danger',
            'recommendation': 'Te recomendamos consultar con un profesional de salud.',
            'icon': '🏥'
        }

# =============================================================================
# RUTAS DE LA APLICACIÓN
# =============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Realizar predicción"""
    try:
        # Verificar que el modelo esté cargado
        if model is None:
            return jsonify({
                'error': 'Modelo no disponible. Por favor, contacta al administrador.'
            }), 500
        
        # Obtener datos del formulario
        data = {
            'Age': float(request.form.get('age', 30)),
            'Gender': request.form.get('gender', 'Male'),
            'Weight (kg)': float(request.form.get('weight', 70)),
            'Height (m)': float(request.form.get('height', 1.75)),
            'Workout_Frequency (days/week)': int(request.form.get('workout_frequency', 3)),
            'Calories': float(request.form.get('calories', 2000)),
            'Proteins': float(request.form.get('proteins', 100)),
            'Carbs': float(request.form.get('carbs', 200)),
            'Fats': float(request.form.get('fats', 60)),
            'Workout_Type': request.form.get('workout_type', 'Cardio'),
            'Experience_Level': int(request.form.get('experience_level', 2)),
            'Resting_BPM': float(request.form.get('resting_bpm', 70)),
            'Avg_BPM': float(request.form.get('avg_bpm', 120)),
            'Session_Duration (hours)': float(request.form.get('session_duration', 1)),
            'Calories_Burned': float(request.form.get('calories_burned', 300)),
        }
        
        # Preprocesar datos
        df_processed = preprocess_input(data)
        
        # Hacer predicción
        prediction = model.predict(df_processed)[0]
        prediction = round(float(prediction), 2)
        
        # Categorizar resultado
        result_info = categorize_fat_percentage(prediction)
        
        # Preparar respuesta
        response = {
            'success': True,
            'fat_percentage': prediction,
            'category': result_info['category'],
            'color': result_info['color'],
            'recommendation': result_info['recommendation'],
            'icon': result_info['icon'],
            'bmi': round(data['Weight (kg)'] / (data['Height (m)'] ** 2), 2)
        }
        
        return render_template('result.html', result=response, input_data=data)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicciones (JSON)"""
    try:
        if model is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        data = request.get_json()
        df_processed = preprocess_input(data)
        prediction = model.predict(df_processed)[0]
        result_info = categorize_fat_percentage(float(prediction))
        
        return jsonify({
            'success': True,
            'fat_percentage': round(float(prediction), 2),
            'category': result_info['category'],
            'recommendation': result_info['recommendation']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """Página sobre el proyecto"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check para Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# =============================================================================
# EJECUTAR APLICACIÓN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)