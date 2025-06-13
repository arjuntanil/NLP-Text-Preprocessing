import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize Stemmer and Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Get user input
text = input("Enter your text: ")

# -------------------------
# Tokenization
# -------------------------
tokens = word_tokenize(text)
print("\nTokenized Words:")
print(tokens)


stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nAfter Stop Word Removal:")
print(filtered_tokens)

# -------------------------
# Stemming
# -------------------------
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\nAfter Stemming:")
print(stemmed_words)

# -------------------------
# Lemmatization using WordNetLemmatizer
# -------------------------
lemmatized_words_nltk = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nAfter Lemmatization (NLTK):")
print(lemmatized_words_nltk)

# -------------------------
# Lemmatization using spaCy
# -------------------------
doc = nlp(text)
lemmatized_words_spacy = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
print("\nAfter Lemmatization (spaCy):")
print(lemmatized_words_spacy)
