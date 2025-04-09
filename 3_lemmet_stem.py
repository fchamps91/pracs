# --- Lemmatization using spaCy ---
nlp = spacy.load("en_core_web_sm")
words = ["cats", "studying", "hiring", "fire"]
lemmatized_words = [token.lemma_ for word in words for token in nlp(word)]
print("\n--- spaCy Lemmatization ---")
print("Original words:", words)
print("Lemmatized words:", lemmatized_words)
print("Number of lemmatized tokens:", len(lemmatized_words))

















































# --- Stemming in spaCy (using NLTK's PorterStemmer) ---
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
text = "Cats are running, studying, and flying in the sky."
doc = nlp(text)
tokens_spacy = [token.text for token in doc]
stemmed_words = [stemmer.stem(token) for token in tokens_spacy]
print("\n--- Stemming using NLTK's PorterStemmer on spaCy tokens ---")
print("Original tokens:", tokens_spacy)
print("Number of tokens:", len(tokens_spacy))
print("Stemmed tokens:", stemmed_words)













# --- Import Packages and Libraries ---
import nltk
# Note: The original file used 'punkt_tab' causing an error; we use 'punkt' instead.
nltk.download('punkt')
nltk.download('wordnet')
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt

# --- Sample Input Data ---
Sample = ("Word Embeddings are numeric representations of words in a lower-dimensional space, "
          "capturing semantic and syntactic information. They play a vital role in Natural Language "
          "Processing (NLP) tasks. This article explores traditional and neural approaches, such as "
          "TF-IDF, Word2Vec, and GloVe, offering insights into their advantages and disadvantages. "
          "Understanding the importance of pre-trained word embeddings, providing a comprehensive "
          "understanding of their applications in various NLP scenarios.")

# --- Tokenization ---
Sample_tokens = word_tokenize(Sample)
print("\n--- Tokenized Words ---")
print(Sample_tokens)
print("\nType of tokens list:", type(Sample_tokens))  # Expected: <class 'list'>
print("Number of tokens:", len(Sample_tokens))         # Expected: 77

# --- Frequency Distribution ---
fdist = FreqDist(Sample_tokens)
print("\n--- Top 5 Most Common Tokens ---")
top5 = fdist.most_common(5)
print(top5)  # Expected: [(',', 6), ('and', 4), ('.', 4), ('of', 3), ('in', 3)]
print("List of all unique tokens:")
print(list(fdist.keys()))

# --- Plotting Frequency Distribution ---
print("\n--- Plotting Token Frequency ---")
fdist.plot(5, cumulative=False)
plt.show()

# --- Stemming using PorterStemmer ---
pst = PorterStemmer()
stemming_results = (pst.stem("Giving"), pst.stem("Buying"), pst.stem("Studying"))
print("\n--- Porter Stemming Examples ---")
print("Stemming of 'Giving', 'Buying', and 'Studying':", stemming_results)

# --- Lemmatization using WordNetLemmatizer ---
lemmatizer = WordNetLemmatizer()
word_to_stem = ["cats", "cacti", "geese", "children"]
print("\n--- Lemmatization Examples ---")
for word in word_to_stem:
    lemma = lemmatizer.lemmatize(word)
    print(f"Word: {word} | Lemmatized: {lemma}")