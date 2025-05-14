import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train_model(self, dataset):
        # Ruta de los datos procesados
        processed_data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/processed/"

        # Cargar los datos procesados
        data = pd.read_csv(f"{processed_data_path}/processed_phishing_email_part_0.csv")  # Ajusta el nombre del archivo
        # Asegúrate de que las columnas sean correctas
        # 'cleaned_text' es el texto procesado y 'label' es la etiqueta (phishing o legítimo)

        # Vectorización con TF-IDF
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(data['cleaned_text'])  # Texto procesado
        y = data['label']  # Etiquetas (0 = legítimo, 1 = phishing)

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo (SVM)
        self.model = SVC(kernel='linear', random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Guardar el modelo y el vectorizador
        joblib.dump(self.model, "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl")
        joblib.dump(self.vectorizer, "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/vectorizer.pkl")

    def predict(self, input_data):
        # Code to classify a given email or URL
        pass