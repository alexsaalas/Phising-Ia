# src/detector/utils.py
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
import nltk
from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('stopwords')

stopwords_es = set(stopwords.words('spanish'))
spell = SpellChecker(language='es')
urgent_keywords = ['urgente', 'inmediato', 'importante', 'atención', 'alerta', 'aviso', 'pago', 'bloqueo', 'cuenta', 'verificar', 'premio', 'ganador', 'banco']

def extract_features(text):
    # Limpieza y tokenización
    words = text.lower().split()
    num_words = len(words)
    num_unique_words = len(set(words))
    num_stopwords = sum(1 for w in words if w in stopwords_es)
    num_links = text.count('http')
    num_unique_domains = len(set([w.split('/')[2] for w in words if w.startswith('http')]))
    num_email_addresses = text.count('@')
    num_spelling_errors = len([w for w in words if w not in stopwords_es and w not in spell and w.isalpha()])
    num_urgent_keywords = sum(1 for w in words if w in urgent_keywords)
    return [num_words, num_unique_words, num_stopwords, num_links, num_unique_domains, num_email_addresses, num_spelling_errors, num_urgent_keywords]

def extract_features_subject(subject):
    words = subject.lower().split()
    num_words = len(words)
    num_unique_words = len(set(words))
    num_urgent_keywords = sum(1 for w in words if w in urgent_keywords)
    num_exclamations = subject.count('!')
    num_uppercase = sum(1 for w in words if w.isupper())
    contains_url = int('http' in subject or 'www' in subject)
    contains_email = int('@' in subject)
    return [num_words, num_unique_words, num_urgent_keywords, num_exclamations, num_uppercase, contains_url, contains_email]