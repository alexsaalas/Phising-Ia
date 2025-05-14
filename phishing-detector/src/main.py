# File: /phishing-detector/phishing-detector/src/main.py

from detector.model import PhishingDetector
from data.preprocess import load_data

def main():
    # Load and preprocess the dataset
    dataset = load_data('data/raw/dataset.csv')
    
    # Initialize the phishing detector
    detector = PhishingDetector()
    
    # Train the model
    detector.train_model(dataset)
    
    # User input for classification
    while True:
        user_input = input("Enter an email or URL to classify (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        prediction = detector.predict(user_input)
        print(f'The input is classified as: {prediction}')

if __name__ == "__main__":
    main()