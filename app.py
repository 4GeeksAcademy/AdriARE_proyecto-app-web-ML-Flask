from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar modelo, encoders y scaler
model = joblib.load('obesity_model.pkl')
le_dict = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Columnas categóricas y numéricas según X_train real
cat_cols = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

num_cols = [
    'Age',
    'Height',
    'Weight',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]

# Orden exacto de columnas del modelo
model_cols = [
    'Gender',
    'Age',
    'Height',
    'Weight',
    'family_history_with_overweight',
    'FAVC',
    'FCVC',
    'NCP',
    'CAEC',
    'SMOKE',
    'CH2O',
    'SCC',
    'FAF',
    'TUE',
    'CALC',
    'MTRANS'
]

# Opciones para el formulario (valores originales)
options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        features = {}

        # Procesar categóricas usando sus LabelEncoders
        for col in cat_cols:
            raw_val = request.form[col]
            features[col] = le_dict[col].transform([raw_val])[0]

        # Procesar numéricas
        for col in num_cols:
            features[col] = float(request.form[col])

        # Crear DataFrame
        df_input = pd.DataFrame([features])

        # Escalar solo numéricas (una sola vez)
        df_input[num_cols] = scaler.transform(df_input[num_cols])

        # Ordenar columnas exactamente como en el modelo
        df_input = df_input[model_cols]

        # Predicción
        pred = model.predict(df_input)[0]

        # Decodificar etiqueta final
        prediction = le_dict['NObeyesdad'].inverse_transform([pred])[0]

    return render_template('index.html', options=options, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)