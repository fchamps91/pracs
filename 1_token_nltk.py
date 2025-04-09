import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.chunk import ne_chunk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Download necessary NLTK resources (if not already downloaded)
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
# nltk.download('stopwords')

# Tokenization Example
Sample = "Neuro- Linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinderin California, United States, in the 1970's. NLP's creators claim there is a connection between newurological processes (neuro), language (linguistuc), and behavioural paterns learnid through experience (programming), and that these can be changed to achieve specific goals in life."
Sample_tokens = word_tokenize(Sample)
print("Tokens:", Sample_tokens)

# Type and Length of Tokens
print("Type:", type(Sample_tokens), "Length:", len(Sample_tokens))

# Frequency of Words
fdisk = FreqDist(Sample_tokens)
top_5 = fdisk.most_common(10)
print("Top 5 Most Common Tokens:", top_5)

# Plotting Token Frequency
fdisk.plot(70, cumulative=False)
plt.show()

word_features = list(fdisk.keys())[:1000]
print("Word Features:", word_features)

# POS Tagging Example
txt = """Sukanya, Rajib and Naba are my good friends.
Sukanya is getting married next year.
Marriage is a big step in one’s life.
It is both exciting and frightening.
But friendship is a sacred bond between people.
It is a special kind of love between us.
Many of you must have tried searching for a friend but never found the right one."""

stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(txt)
for i in tokenized:
    wordsList = word_tokenize(i)
    wordsList = [w for w in wordsList if not w in stop_words]
    tagged = nltk.pos_tag(wordsList)
    print("POS Tags:", tagged)