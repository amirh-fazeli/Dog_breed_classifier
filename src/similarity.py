# src/similarity.py
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Define the breeds you are actually working with
AVAILABLE_BREEDS = [
    "Golden Retriever",
    "Labrador Retriever",
    "German Shepherd",
    "Shih Tzu",
    "Beagle",
    "Siberian Husky",
    "Pomeranian",
    "Yorkshire Terrier",
    "Doberman",
    "Boxer"
]

with open("data/dog_breeds.jsonld", "r", encoding="utf-8") as f:
    data = json.load(f)

main_entities = data.get("mainEntity", [])

# Extract features into DataFrame
records = []
for entry in main_entities:
    breed = entry["name"]
    if breed not in AVAILABLE_BREEDS:
        continue  # skip breeds not in your project
    props = {prop["name"]: prop["value"] for prop in entry.get("additionalProperty", [])}
    record = {"breed": breed}
    # numeric attributes
    for attr in ["affectionLevel", "playfulness", "energyLevel", "intelligence",
                 "trainability", "sheddingLevel", "grooming", "childFriendly",
                 "otherDogsFriendly"]:
        record[attr] = int(props.get(attr, 0))  # default 0 if missing

    # categorical attributes
    record["size"] = props.get("size", "")
    record["coatType"] = props.get("coatType", "")

    records.append(record)

df = pd.DataFrame(records).set_index("breed")

# --- Clean list-valued fields ---
list_fields = ["size", "coatType"]

for col in list_fields:
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: "_".join(x) if isinstance(x, list) else x
        )

df_encoded = pd.get_dummies(df, columns=list_fields)

# Compute similarity matrix
similarity_matrix = cosine_similarity(df_encoded)

def get_similar_breeds(breed_name, top_n=3):
    if breed_name not in df_encoded.index:
        return []
    idx = df_encoded.index.get_loc(breed_name)
    sims = similarity_matrix[idx]
    similar_indices = sims.argsort()[-top_n-1:-1][::-1]  # skip itself
    return df_encoded.index[similar_indices].tolist()
