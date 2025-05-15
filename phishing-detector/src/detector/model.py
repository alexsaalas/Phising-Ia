import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os
import glob

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_model(self, dataset=None):
        processed_data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/processed/"
        files = glob.glob(os.path.join(processed_data_path, "processed_phising_email*.csv"))
        dataframes = []
        for f in files:
            df = pd.read_csv(f)
            # Si no tiene cabecera, pandas pone columnas 0 y 1
            if 'text' not in df.columns:
                # Si solo hay dos columnas, asume texto y etiqueta
                if len(df.columns) == 2:
                    df.columns = ['text', 'label']
                else:
                    raise ValueError(f"Archivo {f} no tiene columnas reconocidas.")
            dataframes.append(df)
        data = pd.concat(dataframes, ignore_index=True)
        print(f"Total filas combinadas: {len(data)}")
        # Eliminar filas vacías
        data = data.dropna(subset=['text'])
        data = data[data['text'].astype(str).str.strip() != '']
        data = data[data['label'].isin([0, 1, '0', '1', 1.0, 0.0])]
        print(f"Filas después de limpiar: {len(data)}")

        # Vectorización con TF-IDF
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(data['text'])

        # Convertir etiquetas a enteros
        y = data['label'].astype(float).astype(int)

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo (SVM)
        self.model = SVC(kernel='linear', random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Guardar el modelo y el vectorizador
        model_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl"
        vectorizer_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/vectorizer.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def predict(self, input_data):
        # Cargar modelo y vectorizador si no están cargados
        if self.model is None or self.vectorizer is None:
            self.model = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl")
            self.vectorizer = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/vectorizer.pkl")

        # Transformar el texto de entrada
        input_vector = self.vectorizer.transform([input_data])
        prediction = self.model.predict(input_vector)

        return prediction[0]

