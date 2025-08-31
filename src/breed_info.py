import json
import re
import os
import pycountry

with open("data/dog_breeds.jsonld", "r", encoding="utf-8") as f:
    data = json.load(f)

main_entities = data.get("mainEntity", [])

# Mapping from predict.py breed names to JSON breed names
BREED_NAME_MAP = {
    "Labrador_Retriever": "Labrador Retriever",
    "German_Shepherd": "German Shepherd",
    "Golden_Retriever": "Golden Retriever",
    "Boxer": "Boxer",
    "Beagle": "Beagle",
    "Pomeranian": "Pomeranian",
    "Siberian_Husky": "Siberian Husky",
    "Doberman": "Doberman Pinscher",
    "Shih-Tzu": "Shih Tzu",
    "Yorkshire_Terrier": "Yorkshire Terrier",
}

FLAGS_DIR = "data/flags"  # folder with SVG files

def clean_attribute_name(name):
    """Remove parentheses from description text."""
    return re.sub(r"\s*\(.*?\)", "", name).strip()

def get_top_attributes(additional_properties, top_n=3):
    """Extract top N numeric attributes sorted by value."""
    numeric_props = []
    for prop in additional_properties:
        value = prop.get("value")
        if isinstance(value, (int, float)):
            name = prop.get("description") or prop.get("name")
            numeric_props.append((clean_attribute_name(name), value))
    numeric_props.sort(key=lambda x: x[1], reverse=True)
    return numeric_props[:top_n]

def country_to_dict(country_field):
    """
    Convert country names (string or list) to {country: flag_path}.
    """
    if not country_field:
        return {}

    if isinstance(country_field, str):
        countries = [c.strip() for c in country_field.split(",")]
    elif isinstance(country_field, list):
        countries = country_field
    else:
        countries = []

    result = {}
    for country in countries:
        try:
            # Normalize common edge cases
            if country.lower() == "england":
                code, name = "GB", "England"
            elif country.lower() == "scotland":
                code, name = "GB", "Scotland"
            elif country.lower() == "usa":
                code, name = "US", "United States"
            else:
                country_obj = pycountry.countries.lookup(country)
                code, name = country_obj.alpha_2, country_obj.name

            filename = f"{code.lower()}.svg"
            filepath = os.path.join(FLAGS_DIR, filename)

            if os.path.exists(filepath):
                result[name] = filepath
            else:
                result[name] = ""  # fallback if flag not available
        except LookupError:
            continue

    return result

def get_breed_info(predicted_breed_name):
    """
    Main hub function.
    Input: breed name from predict.py
    Output: dictionary with breed info
    """
    json_breed_name = BREED_NAME_MAP.get(predicted_breed_name)
    if not json_breed_name:
        return {"error": f"Breed {predicted_breed_name} not found in mapping."}

    breed_entry = next((entry for entry in main_entities if entry.get("name") == json_breed_name), None)
    if not breed_entry:
        return {"error": f"Breed {json_breed_name} not found in JSON data."}

    additional_props = breed_entry.get("additionalProperty", [])

    top_attrs = get_top_attributes(additional_props, top_n=3)

    country_of_origin = next((prop["value"] for prop in additional_props if prop.get("name") == "countryOfOrigin"), "")

    info = {
        "breed": breed_entry.get("name"),
        "alternateName": next((prop["value"] for prop in additional_props if prop.get("name") == "alternateName"), ""),
        "countryOfOrigin": country_to_dict(country_of_origin),  # NEW
        "size": next((prop["value"] for prop in additional_props if prop.get("name") == "size"), []),
        "top_attributes": top_attrs
    }

    return info

# Example usage
if __name__ == "__main__":
    example_breed = "Golden_Retriever"
    from pprint import pprint
    pprint(get_breed_info(example_breed))
