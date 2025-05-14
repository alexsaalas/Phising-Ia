import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Rutas de los datos
raw_data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/raw/"
processed_data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/processed/"

# Crear la carpeta de datos procesados si no existe
os.makedirs(processed_data_path, exist_ok=True)

# Función para limpiar texto
def clean_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Unir tokens en un solo string
    return ' '.join(tokens)

# Procesar cada archivo en la carpeta raw
for file in os.listdir(raw_data_path):
    if file.endswith(".csv"):
        print(f"Processing {file}...")
        # Leer el archivo
        data = pd.read_csv(os.path.join(raw_data_path, file))
        
        # Asumimos que hay una columna llamada 'text' con el contenido a procesar
        if 'text' in data.columns:
            data['cleaned_text'] = data['text'].apply(clean_text)
        
        # Guardar el archivo procesado
        processed_file_path = os.path.join(processed_data_path, f"processed_{file}")
        data.to_csv(processed_file_path, index=False)
        print(f"Processed file saved: {processed_file_path}")