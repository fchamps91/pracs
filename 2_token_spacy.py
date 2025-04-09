# Importing spaCy
import spacy

# Creating a blank language object and tokenizing words in the sentence
nlp = spacy.blank("en")
doc = nlp("GeeksforGeeks is a one stop learning destination for geeks.")

for token in doc:
    print(token)

# Parts of Speech (POS) tagging
nlp = spacy.load("en_core_web_sm")

text = """My name is Shaurya Uppal.
I enjoy writing articles on GeeksforGeeks checkout
my other article by going to my profile section."""

doc = nlp(text)

for token in doc:
    print(token, token.pos_)

print("Verbs:", [token.text for token in doc if token.pos_ == "VERB"])