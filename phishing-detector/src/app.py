# src/app.py
from flask import Flask, request, render_template
import joblib
import pandas as pd
from detector.utils import extract_features

app = Flask(__name__)

# Cargar el modelo y el escalador
model = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl")
scaler = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/scaler.pkl")

@app.route('/', methods=['GET'])
def home():
    return(render_template('index.html', prediction=None))

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    features = extract_features(input_text)
    print("Características extraídas:", features)  # <-- Añade esto para depurar
    # Convertir a DataFrame
    feature_columns = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
                       'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 
                       'num_urgent_keywords']
    features_df = pd.DataFrame([features], columns=feature_columns)
    # Escalar las características
    features_scaled = scaler.transform(features_df)
    # Predecir
    prediction = model.predict(features_scaled)[0]
    result = "Phishing" if prediction == 1 else "Legítimo"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)