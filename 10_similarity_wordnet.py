import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')  # Open Multilingual WordNet


# Sample dataset
data = [
    "The cat chased the mouse.",
    "A feline pursued a rodent.",
    "The dog barked loudly.",
    "A canine vocalized with intensity.",
    "The sun is shining brightly.",
    "The day is radiant.",
    "I enjoy reading books.",
    "I find pleasure in perusing literature.",
    "The car is fast.",
    "The automobile has high velocity."
]


def get_wordnet_pos(treebank_tag):
    """Convert treebank part-of-speech tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def calculate_sentence_similarity(sentence1, sentence2):
    """Calculate semantic similarity between two sentences using WordNet."""
    tokens1 = word_tokenize(sentence1.lower())
    tokens2 = word_tokenize(sentence2.lower())
    tagged1 = nltk.pos_tag(tokens1)
    tagged2 = nltk.pos_tag(tokens2)


    synsets1 = [wordnet.synsets(word, pos=get_wordnet_pos(tag)) for word, tag in tagged1]
    synsets2 = [wordnet.synsets(word, pos=get_wordnet_pos(tag)) for word, tag in tagged2]


    synsets1 = [ss[0] if ss else None for ss in synsets1]
    synsets2 = [ss[0] if ss else None for ss in synsets2]


    # Remove None values (words without synsets)
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]


    if not synsets1 or not synsets2:
        return 0.0  # Return 0 if no synsets found


    similarities = []
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)  # Wu-Palmer similarity
            if similarity is not None:
                similarities.append(similarity)


    if not similarities:
        return 0.0  # Return 0 if no similarities found.


    return np.mean(similarities)


# Calculate similarity matrix
num_sentences = len(data)
similarity_matrix = np.zeros((num_sentences, num_sentences))


for i in range(num_sentences):
    for j in range(num_sentences):
        similarity_matrix[i, j] = calculate_sentence_similarity(data[i], data[j])


# Sort similarity matrix
sorted_similarities = []
for i in range(num_sentences):
    for j in range(i + 1, num_sentences):  # Avoid duplicate pairs
        sorted_similarities.append(((i, j), similarity_matrix[i, j]))


# Sort by similarity score in descending order
sorted_similarities = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)


# Print sorted similarity scores
print("Sorted Sentence Similarity Scores (Descending Order):")
for pair, score in sorted_similarities:
    print(f"Sentence {pair[0]} and Sentence {pair[1]}: {score}")


# Example Usage to compare specific sentences.
sentence1_index = 0
sentence2_index = 1
similarity_score = calculate_sentence_similarity(data[sentence1_index], data[sentence2_index])
print(f"\nSimilarity between sentence '{data[sentence1_index]}' and '{data[sentence2_index]}': {similarity_score}")