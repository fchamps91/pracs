# Importing necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
nltk.download('punkt_tab')
# Download the necessary NLTK data
nltk.download('punkt')


# Taking the sample data
Sample = "Natural Language Processing (NLP) is a branch of"


# Applying tokenization
Sample_tokens = word_tokenize(Sample)


# Checking the tokens and their length
print(Sample_tokens)
print("Type:", type(Sample_tokens))
print("Length:", len(Sample_tokens))


# Creating a frequency distribution of the tokens
fdist = FreqDist(Sample_tokens)


# Getting the top 5 most common tokens
top_5 = fdist.most_common(5)
print("Top 5 tokens:", top_5)


# Plotting the frequency distribution
fdist.plot(title="Token Frequency Distribution", cumulative=False)


# Optional: if you want to specifically plot the top 5
top_5_tokens = [item[0] for item in top_5]
top_5_counts = [item[1] for item in top_5]


plt.bar(top_5_tokens, top_5_counts)
plt.title("Top 5 Token Frequencies")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.show()























# listing 1888 most frequent words

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt


# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('gutenberg')


# Load a sample large text corpus (e.g., "Emma" by Jane Austen)
from nltk.corpus import gutenberg
sample_text = gutenberg.raw('austen-emma.txt')


# Tokenize the text
tokens = word_tokenize(sample_text)


# Create a frequency distribution of the tokens
fdist = FreqDist(tokens)


# Get the top 1888 most common tokens
top_1888 = fdist.most_common(1888)


# Extract just the words from the tuples
word_features = [word for word, _ in top_1888]


# Print the top 1888 most frequent words
print(word_features)


# Optional: Plot the frequency distribution of the top 30 most common words
fdist.plot(30, cumulative=False)
plt.show()




































# Practical 6: Frequency Distribution


import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download('punkt')

Sample = "Natural Language Processing (NLP) is a branch of"
Sample_tokens = word_tokenize(Sample)
print("\n--- Tokens (Practical 6) ---")
print(Sample_tokens)
print("Token list type:", type(Sample_tokens))
print("Number of tokens:", len(Sample_tokens))

fdist = FreqDist(Sample_tokens)
top_5 = fdist.most_common(5)
print("\n--- Top 5 Tokens (Practical 6) ---")
print(top_5)

fdist.plot(title="Token Frequency Distribution (Practical 6)", cumulative=False)
top_5_tokens = [item[0] for item in top_5]
top_5_counts = [item[1] for item in top_5]
plt.figure()
plt.bar(top_5_tokens, top_5_counts)
plt.title("Top 5 Token Frequencies (Practical 6)")
plt.xlabel("Tokens")
plt.ylabel("Frequency")
plt.show()