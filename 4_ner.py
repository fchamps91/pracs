# --- Named Entity Recognition (NER) using spaCy ---
nlp = spacy.load('en_core_web_sm')
text = """
Apple Inc. is planning to open a new store in Paris by the end of 2025.
Tim Cook, the CEO of Apple, announced this on January 22, 2025.
The event will be held at the Eiffel Tower.
"""
doc = nlp(text)
print("\n--- spaCy NER ---")
entities = []
for ent in doc.ents:
    entities.append((ent.text, ent.label_))
    print(f"Entity: {ent.text} | Label: {ent.label_}")
print("Total entities found:", len(entities))

# Visualize the Named Entities
from spacy import displacy

# Visualize entities in the text
displacy.render(doc, style='ent', jupyter=True)






























# --- Named Entity Recognition using NLTK ---
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
nltk.download('punkt')  # using 'punkt' instead of 'punkt_tab'
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
string = "Bruce Wayne stays in Gotham. He owns Wayne Industries"
string_token = word_tokenize(string)
string_tag = pos_tag(string_token)
net = ne_chunk(string_tag)
print("\n--- NLTK NER ---")
print(net)
print("Number of tokens in NLTK NER input:", len(string_token))



named_entities = []


for subtree in ner_tree:
    if isinstance(subtree, nltk.Tree):
        entity = " ".join([word for word, tag in subtree.leaves()])
        label = subtree.label()
        named_entities.append((entity, label))


  print(named_entities)














