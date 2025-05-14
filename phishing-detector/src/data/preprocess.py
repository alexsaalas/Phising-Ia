def load_dataset(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(filtered_tokens)

def vectorize_data(data, method='tfidf'):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()

    return vectorizer.fit_transform(data)