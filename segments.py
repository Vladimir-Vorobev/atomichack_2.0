import pickle
import pandas as pd

with open('instruction_chunks.pkl', 'rb') as r:
    instruction_chunks = pickle.load(r)

with open('question_embeddings.pkl', 'rb') as r:
    question_embeddings = pickle.load(r)

with open('all_definitions.pkl', 'rb') as r:
    all_definitions = pickle.load(r)
    unique_definitions = []

    for instruction_definitions in all_definitions.values():
        for key, value in instruction_definitions.items():
            unique_definitions.append([key, value])

    unique_definitions = pd.DataFrame(
        unique_definitions, columns=['term', 'definition']
    ).drop_duplicates(subset=['term'], keep=False)

    unique_definitions = {
        term: definition for term, definition in zip(unique_definitions['term'], unique_definitions['definition'])
    }

print(unique_definitions)
