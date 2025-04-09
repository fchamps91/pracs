# --- Generating N-grams using NLTK ---
from nltk.util import ngrams
token_list = word_tokenize("I love natural language processing.")
unigrams = list(ngrams(token_list, 1))
bigrams = list(ngrams(token_list, 2))
trigrams = list(ngrams(token_list, 3))
n = 4
ngrams_list = list(ngrams(token_list, n))
print("\n--- NLTK N-grams ---")
print("Unigrams (count:):", len(unigrams), unigrams)
print("Bigrams (count:):", len(bigrams), bigrams)
print("Trigrams (count:):", len(trigrams), trigrams)
print(f"{n}-grams (count:):", len(ngrams_list), ngrams_list)







































# --- Generating N-grams using spaCy ---
doc = nlp("I love natural language processing.")
spacy_unigrams = [token.text for token in doc]
spacy_bigrams = [f\"{doc[i].text} {doc[i+1].text}\" for i in range(len(doc)-1)]
spacy_trigrams = [f\"{doc[i].text} {doc[i+1].text} {doc[i+2].text}\" for i in range(len(doc)-2)]
n = 4
spacy_ngrams = [" ".join([doc[i+j].text for j in range(n)]) for i in range(len(doc)-n+1)]
print("\n--- spaCy N-grams ---")
print("Unigrams:", spacy_unigrams, "(Count:", len(spacy_unigrams),")")
print("Bigrams:", spacy_bigrams, "(Count:", len(spacy_bigrams),")")
print("Trigrams:", spacy_trigrams, "(Count:", len(spacy_trigrams),")")
print(f"{n}-grams:", spacy_ngrams, "(Count:", len(spacy_ngrams),")")
















# CODE: Unigram, Bigram and Trigram using spaCy.

# !pip install spacy
# !python -m spacy download en_core_web_sm


import spacy
nlp = spacy.load("en_core_web_sm")
text = "I love natural language processing."
doc = nlp(text)

# Generate Unigrams

unigrams = [token.text for token in doc]
print("Unigrams:")
print(unigrams)


# Generate Bigrams

bigrams = [(doc[i].text, doc[i+1].text) for i in range(len(doc)-1)]
print("Bigrams:")
print(bigrams)


# Generate Trigrams

trigrams = [(doc[i].text, doc[i+1].text, doc[i+2].text) for i in range(len(doc)-2)]
print("Trigrams:")
print(trigrams)






