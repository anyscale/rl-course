import requests
import os

API_KEY = os.environ.get("DEEPL_API_KEY")

# Plan of attack:
# 1. copy notebooks to notebooks/locale for each locale (delete old ones on demand)
# 2. For each set of localized notebooks, run translations for cell-sources in-place.
# 3. Generate course material for each locale.

with open("notebooks/01-rl-components/01_03_policies.ipynb", "r") as f:
    text = f.read()


def translate_cell(text):
    result = requests.get(
        "https://api-free.deepl.com/v2/translate",
        params={
            "auth_key": API_KEY,
            "target_lang": "DE",
            "text": text,
            "tag_handling": "html"
        },
    )
    return result.json()["translations"][0]["text"]


translated_text = translate_cell(text)

with open("test.ipynb", "w") as f:
    f.write(translated_text)
