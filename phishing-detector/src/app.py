from google import genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument
from google.cloud import speech
from flask import Flask, request, render_template
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configura tu API KEY de Gemini
# Es crucial que uses una variable de entorno para tu API Key en producción
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
# Aquí la pongo directamente para que funcione al inicio, pero CAMBIA ESTO en producción
# Asegúrate de tener una API Key válida de Gemini 1.5 Flash
# https://makersuite.google.com/
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Mejor usar variable de entorno siempre

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY no configurada. Configura la variable de entorno.")
    # Puedes salir o manejar el error de otra manera
    # exit() # Descomenta para salir si no hay key

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Usamos el modelo que soporta multimodal
    MODEL = 'gemini-1.5-flash-latest' # O 'gemini-1.5-pro-latest'
    # Opcional: Verifica si el modelo existe
    try:
        genai.get_model(MODEL)
    except Exception as e:
         print(f"Advertencia: El modelo '{MODEL}' puede no ser válido o accesible. Error: {e}")
         # Intenta con un modelo conocido que debería funcionar si tienes una key válida
         MODEL = 'gemini-1.5-flash-latest' # O 'gemini-1.5-pro-latest'

except Exception as e:
    print(f"Error configurando Gemini: {e}")
    # Manejar el error de configuración si la API Key es inválida, etc.


# Configura el cliente de Speech-to-Text (asegúrate de que las credenciales de Google Cloud estén configuradas en tu entorno)
try:
    speech_client = speech.SpeechClient()
except Exception as e:
    print(f"Advertencia: No se pudo inicializar el cliente de Speech-to-Text. Los audios no funcionarán. Error: {e}")
    speech_client = None


# Configura la carpeta para subir archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3'} # Añadido 'gif' por si acaso
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Verifica extensiones permitidas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    # Asegúrate de que la carpeta de uploads exista
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return render_template('index.html', prediction=None, reasons=None, input_type=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_type = None
    content_for_gemini = None
    input_text = None # Variable para almacenar texto (directo o transcrito)

    try:
        # Handle text input
        if 'text' in request.form and request.form['text']:
            input_type = 'text'
            input_text = request.form['text']
            # El prompt se construye más abajo para reutilizarlo

        # Handle image upload
        elif 'image' in request.files and request.files['image'].filename:
            input_type = 'image'
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)

                    # Prepare image content for Gemini
                    with open(filepath, 'rb') as img_file:
                        # Read image bytes
                        image_bytes = img_file.read()

                    # Prepare the multimodal content list
                    content_for_gemini = [
                        {"mime_type": file.mimetype, "data": image_bytes},
                        # El texto del prompt se añade como otra parte
                        # {"text": "Tu prompt para analizar la imagen"} # Se añade más abajo
                    ]
                    # El prompt textual se añadirá al final de la lista de contenidos

                finally:
                    # Clean up uploaded file regardless of success or failure reading it
                    if os.path.exists(filepath):
                        os.remove(filepath)

            else:
                return render_template('index.html', prediction="Formato de imagen no permitido.", reasons=None, input_type=input_type)

        # Handle audio upload
        elif 'audio' in request.files and request.files['audio'].filename:
            input_type = 'audio'
            if not speech_client:
                 return render_template('index.html', prediction="El servicio Speech-to-Text no está disponible.", reasons=None, input_type=input_type)

            file = request.files['audio']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)

                    # Transcribe audio to text
                    with open(filepath, 'rb') as audio_file:
                        audio_content = audio_file.read()

                    # Determine encoding based on file extension (basic check)
                    if filename.lower().endswith('.wav'):
                         encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
                    elif filename.lower().endswith('.mp3'):
                         encoding = speech.RecognitionConfig.AudioEncoding.MP3
                    else:
                         # Fallback or specific handling for other types if needed
                         encoding = speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED # Let API guess

                    audio = speech.RecognitionAudio(content=audio_content)
                    config = speech.RecognitionConfig(
                        encoding=encoding,
                        language_code='es-ES',  # Ajusta para tu idioma
                        # sample_rate_hertz=... # Opcional: especifica si conoces la tasa de muestreo
                    )
                    print("Iniciando transcripción de audio...")
                    # La transcripción puede ser síncrona para archivos pequeños/medianos
                    # Para archivos grandes, considera async (no cubierto aquí)
                    response = speech_client.recognize(config=config, audio=audio)
                    print("Transcripción completada.")

                    if not response.results:
                         input_text = "(No se pudo transcribir el audio)"
                    else:
                         input_text = ''.join(result.alternatives[0].transcript for result in response.results)
                         print(f"Texto transcrito: {input_text}")

                    # The prompt will be constructed using input_text later

                except Exception as e:
                    print(f"Error durante la transcripción de audio: {e}")
                    input_text = f"(Error al transcribir audio: {e})" # Report error in text

                finally:
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)

            else:
                return render_template('index.html', prediction="Formato de audio no permitido.", reasons=None, input_type=input_type)

        else:
            return render_template('index.html', prediction="No se proporcionó texto, imagen o audio.", reasons=None, input_type=None)

        # --- Prepare the final prompt and content for Gemini ---

        # Define the base prompt structure
        base_prompt = (
            "Eres un detector de phishing. Analiza el siguiente {input_description} y clasifícalo como 'Phishing' o 'Legítimo'. "
            "Si es 'Phishing', proporciona una lista de razones específicas por las que es sospechoso. "
            "Si es 'Legítimo', no es necesario proporcionar razones. "
            "Formatea la respuesta exactamente así:\n"
            "Clasificación: [Phishing o Legítimo]\n"
            "Razones:\n"
            "- [Razón 1, si es Phishing]\n"
            "- [Razón 2, si es Phishing]\n"
            "(Si es Legítimo, deja 'Razones' vacío)\n\n"
        )

        if input_type == 'text':
             prompt_text = base_prompt.format(input_description="texto") + f"Texto a analizar:\n{input_text}\n\nRespuesta:"
             content_for_gemini = prompt_text # For text-only, content is just the prompt string
        elif input_type == 'audio':
             prompt_text = base_prompt.format(input_description="texto transcrito de un audio") + f"Texto a analizar:\n{input_text}\n\nRespuesta:"
             content_for_gemini = prompt_text # For audio, we send the transcribed text as text
        elif input_type == 'image':
             prompt_text = base_prompt.format(input_description="imagen (que puede ser una captura de pantalla de un correo, sitio web, o mensaje)") + "Respuesta:"
             # For image, content_for_gemini is already a list [image_part]. Add the text prompt part.
             content_for_gemini.append({"text": prompt_text})

        # --- Call Gemini API ---
        if content_for_gemini is not None:
            print(f"Enviando a Gemini (Tipo: {input_type})...")
            try:
                # For multimodal inputs (like image), counting tokens explicitly before the call
                # can be complex/different. Relying on catching ResourceExhausted is often
                # simpler unless specific token limits per part are needed.
                # For text/audio (which become text), we can still count if desired, but
                # the prompt format might make it unnecessary unless the input_text is huge.
                # Let's skip the explicit count for now and rely on exception handling.

                response = genai.GenerativeModel(MODEL).generate_content(
                    contents=content_for_gemini # Use the prepared content
                    # stream=True # Optional: if you want to stream the response
                )
                print("Respuesta de Gemini recibida.")

                # Handle potential blocked content or empty responses
                if not response.candidates:
                     return render_template('index.html', prediction="La respuesta de Gemini no contiene candidatos (posiblemente contenido bloqueado).", reasons=None, input_type=input_type)
                if not response.candidates[0].content.parts:
                     return render_template('index.html', prediction="La respuesta de Gemini está vacía.", reasons=None, input_type=input_type)


                result_text = response.candidates[0].content.parts[0].text.strip()
                print(f"Resultado crudo de Gemini:\n{result_text}")

                # Parse the response
                classification = "Desconocido" # Default
                reasons = []
                lines = result_text.split('\n')
                is_reasons_section = False
                for line in lines:
                    line = line.strip()
                    if line.startswith("Clasificación:"):
                        classification = line.replace("Clasificación:", "").strip()
                        # Normalize classification to expected values
                        if classification.lower() == 'phishing':
                            classification = 'Phishing'
                        elif classification.lower() == 'legitimo' or classification.lower() == 'legítimo':
                            classification = 'Legítimo'
                        else:
                             classification = 'Desconocido' # Handle unexpected classification

                    elif line == "Razones:":
                         is_reasons_section = True
                    elif is_reasons_section and line.startswith("- "):
                        reasons.append(line[2:].strip())
                    elif is_reasons_section and not line and not reasons:
                        # Handle case where "Razones:" is present but followed by nothing
                        pass # Keep reasons list empty
                    elif is_reasons_section and line and not line.startswith("- "):
                         # Stop reading reasons if format changes unexpectedly after "Razones:"
                         is_reasons_section = False


                # If classification is Legítimo, ensure reasons list is empty
                if classification == 'Legítimo':
                    reasons = []
                # If classification is Phishing but no reasons were found, add a default message
                elif classification == 'Phishing' and not reasons:
                     reasons = ["No se encontraron razones específicas en la respuesta del modelo, pero fue clasificado como Phishing."]
                 # If classification is Unknown and there's response text, maybe add it as a reason?
                elif classification == 'Desconocido':
                     reasons = [f"El modelo no proporcionó una clasificación clara. Respuesta completa: {result_text}"]


                return render_template('index.html', prediction=classification, reasons=reasons if reasons else None, input_type=input_type)

            except ResourceExhausted as e:
                 print(f"Error ResourceExhausted: {e}")
                 return render_template('index.html', prediction="Límite de cuota excedido, intenta de nuevo más tarde.", reasons=None, input_type=input_type)
            except InvalidArgument as e:
                 print(f"Error InvalidArgument: {e}")
                 return render_template('index.html', prediction=f"Error en la entrada para Gemini (InvalidArgument): {e}. Asegúrate de que la API Key es correcta y el modelo '{MODEL}' es accesible.", reasons=None, input_type=input_type)
            except Exception as e:
                 print(f"Error llamando a Gemini: {e}")
                 return render_template('index.html', prediction=f"Error al procesar con Gemini: {str(e)}", reasons=None, input_type=input_type)
        else:
             # Should not happen if input_type is set, but as a fallback
             return render_template('index.html', prediction="Error interno: No se pudo preparar el contenido para el modelo.", reasons=None, input_type=input_type)


    except Exception as e:
        print(f"Error general en /predict: {e}")
        return render_template('index.html', prediction=f"Error inesperado: {str(e)}", reasons=None, input_type=input_type)

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Ensure your Google Cloud credentials are set up for Speech-to-Text if you use audio
    # Example: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
    # Or configure them via environment variables or default locations
    # print("Asegúrate de tener las credenciales de Google Cloud configuradas para Speech-to-Text.")
    print(f"Usando modelo Gemini: {MODEL}")
    if GEMINI_API_KEY == os.getenv("YOUR_API_KEY"):
         print("¡ADVERTENCIA! Estás usando la API Key por defecto. Configura la variable de entorno GEMINI_API_KEY.")
    app.run(debug=True)