from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
doc = nlp("2018 FIFA world cup: France won!!!")
pattern = [{'IS_DIGIT': True}, {'LOWER': 'fifa'}, {'LOWER': 'world'}, {'LOWER': 'cup'}]
matcher = Matcher(nlp.vocab)
matcher.add('fifa_pattern', [pattern])
matches = matcher(doc)
print("\n--- Pattern Matching ---")
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print("Matched Span:", matched_span.text)
print("Total matches found:", len(matches))


























import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
doc = nlp("2018 FIFA World Cup France won!!!")
pattern = [
    {'is_digit': True},  # Match digits (e.g., 2018)
    {'lower': 'fifa'},  # Match the word 'fifa' in lowercase
    {'lower': 'world'},  # Match the word 'world' in lowercase
    {'lower': 'cup'}     # Match the word 'cup' in lowercase
]
matcher = Matcher(nlp.vocab)
matcher.add('fifa_pattern', [pattern])
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]  # Get the span from the document
    print(matched_span)  # Print the matched span
