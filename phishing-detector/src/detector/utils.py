def tokenize(text):
    # Tokenizes the input text into words
    return text.split()

def remove_stopwords(tokens, stopwords):
    # Removes stopwords from the list of tokens
    return [token for token in tokens if token not in stopwords]

def preprocess_text(text, stopwords):
    # Preprocesses the input text by tokenizing and removing stopwords
    tokens = tokenize(text)
    return remove_stopwords(tokens, stopwords)