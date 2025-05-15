import pandas as pd
import os

processed_data_path = "C:/Users/alexs/Documents/GitHub/Phising-Ia/phishing-detector/data/processed/"
data_file = "processed_CEAS_08.csv"  # Cambia por el archivo que quieras revisar

data = pd.read_csv(os.path.join(processed_data_path, data_file))
print("Primeras filas:")
print(data.head())
print("\nCantidad de NaN en 'body':", data['body'].isna().sum())
print("Cantidad total de filas:", len(data))
print("Primeros 10 valores de 'body':")
print(data['body'].head(10))