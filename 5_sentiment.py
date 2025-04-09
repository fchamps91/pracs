import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer

# Dataset with text and corresponding review sentiment
dataset = [
    ["I liked the movie", "positive"],
    ["It's a good movie. Nice story", "positive"],
    ["Hero's acting is bad but heroine looks good. Overall nice movie", "positive"],
    ["Nice songs. But sadly boring ending.", "negative"],
    ["sad movie, boring movie", "negative"]
]

# Create a pandas DataFrame
df = pd.DataFrame(dataset, columns=["Text", "Reviews"])

# Initialize an empty list for the corpus
corpus = []

# Preprocess text data in the DataFrame
for i in range(0, len(df)):
    text = re.sub('[^a-zA-Z]', ' ', df['Text'][i])  # Remove non-alphabetical characters
    text = text.lower()  # Convert text to lowercase
    text = text.split()  # Split text into tokens
    ps = PorterStemmer()  # Initialize the Porter Stemmer
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]  # Apply stemming and remove stopwords
    text = " ".join(text)  # Combine tokens into a single string
    corpus.append(text)

# Convert text into a feature matrix
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()  # Feature matrix
y = df.iloc[:, 1].values  # Labels (positive/negative reviews)

# Print outputs
print(X)  # Feature matrix
print(y)  # Labels
print(corpus)  # Processed text corpus
print(df)  # Original DataFrame












































import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
from sklearn.feature_extraction.text import CountVectorizer

dataset = [["I liked the movie", "positive"],
           ["It's a good movie. Nice story", "positive"],
           ["Hero's acting is bad but heoirne looks good. Overall nice movie", "positive"],
           ["Nice songs. But sadly boring ending.", "negative"],
           ["sad movie, boring movie", "negative"]]

df = pd.DataFrame(dataset, columns=["Text", "Reviews"])
corpus = []
for i in range(len(df)):
    text = re.sub('[^a-zA-Z]', ' ', df.loc[i, 'Text'])
    text = text.lower()
    tokens = text.split()
    ps = PorterStemmer()
    filtered_tokens = [ps.stem(word) for word in tokens if word not in set(stopwords.words('english'))]
    processed_text = " ".join(filtered_tokens)
    corpus.append(processed_text)

print("\n--- Processed Corpus ---")
print(corpus)
print("Total reviews processed:", len(corpus))

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

print("\n--- Feature Matrix (X) Shape ---")
print(X.shape)
print("\n--- Review Labels (y) ---")
print(y)
print("\n--- DataFrame ---")
print(df)