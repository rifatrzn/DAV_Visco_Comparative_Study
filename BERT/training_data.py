import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(query):
    # Process query with spaCy
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# Example: Extract entities from a query
query = "Sales revenue in Q1 2021 for product category electronics."
entities = extract_entities(query)
print(f"Extracted entities: {entities}")