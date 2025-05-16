from google import genai
from google.genai import types
from flask import Flask, request, render_template

app = Flask(__name__)

# Configura tu API KEY de Gemini
GEMINI_API_KEY = "AIzaSyDYVXUL6WcREkDzU-lW44Kz-AtXS3CPoho"
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