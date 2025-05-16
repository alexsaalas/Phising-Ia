import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from flask import Flask, request, render_template

load_dotenv()  # Carga las variables del .env

app = Flask(__name__)

# Obtén la API KEY de Gemini desde el .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1')
)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    prompt = (
        "Eres un detector de phishing. Analiza el siguiente texto y responde solo con 'Phishing' o 'Legítimo':\n\n"
        f"{input_text}\n\n"
        "Respuesta:"
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash-preview-04-17',
        contents=prompt
    )
    result = response.text.strip()
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)