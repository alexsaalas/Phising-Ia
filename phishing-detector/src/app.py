from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl")
vectorizer = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Página principal con el formulario

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto ingresado por el usuario
    input_text = request.form['text']
    
    # Vectorizar el texto
    input_vectorized = vectorizer.transform([input_text])
    
    # Realizar la predicción
    prediction = model.predict(input_vectorized)[0]
    
    # Interpretar el resultado
    result = "Phishing" if prediction == 1 else "Legítimo"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)