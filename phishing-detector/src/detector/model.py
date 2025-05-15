import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train_model(self, dataset=None):
        # Define the data path (adjust to your raw folder)
        data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/raw/"
        dataframes = []

        # Load dataset(s)
        if dataset:
            if isinstance(dataset, str):
                files = [os.path.join(data_path, dataset)]
            elif isinstance(dataset, list):
                files = [os.path.join(data_path, f) for f in dataset]
            else:
                raise ValueError("dataset must be str or list of str")
        else:
            files = glob.glob(os.path.join(data_path, "*.csv"))

        for f in files:
            df = pd.read_csv(f)
            # Verify required columns
            required_columns = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
                                'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 
                                'num_urgent_keywords', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Archivo {f} no tiene todas las columnas requeridas: {required_columns}")
            dataframes.append(df)

        if not dataframes:
            raise ValueError("No se encontraron archivos de datos para entrenar.")

        # Combine dataframes
        data = pd.concat(dataframes, ignore_index=True)
        print(f"Total filas combinadas: {len(data)}")

        # Clean data
        data = data.dropna()  # Remove rows with missing values
        data = data[data['label'].isin([0, 1, '0', '1', 0.0, 1.0])]  # Ensure valid labels
        print(f"Filas después de limpiar: {len(data)}")

        # Define features and labels
        feature_columns = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
                           'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 
                           'num_urgent_keywords']
        X = data[feature_columns]
        y = data['label'].astype(float).astype(int)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model (SVM)
        self.model = SVC(kernel='linear', random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save the model and scaler
        model_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl"
        scaler_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/scaler.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def predict(self, input_data):
        if self.model is None or self.scaler is None:
            self.model = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/phishing_model.pkl")
            self.scaler = joblib.load("C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/src/detector/scaler.pkl")
        
        # Ensure input_data is a DataFrame with the correct feature columns
        feature_columns = ['num_words', 'num_unique_words', 'num_stopwords', 'num_links', 
                           'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 
                           'num_urgent_keywords']
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=feature_columns)
        elif not all(col in input_data.columns for col in feature_columns):
            raise ValueError(f"input_data debe contener las columnas: {feature_columns}")

        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        return prediction[0]

# Example usage
if __name__ == "__main__":
    detector = PhishingDetector()
    detector.train_model(dataset="email_phishing_data.csv")
    
    # Example prediction
    sample_input = {
        'num_words': 100,
        'num_unique_words': 80,
        'num_stopwords': 30,
        'num_links': 2,
        'num_unique_domains': 1,
        'num_email_addresses': 1,
        'num_spelling_errors': 5,
        'num_urgent_keywords': 1
    }
    prediction = detector.predict(sample_input)
    print(f"Predicción para el ejemplo: {'Phishing' if prediction == 1 else 'No Phishing'}")