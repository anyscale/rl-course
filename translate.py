import os
import nbformat
import requests

from utils import get_notebook, get_notebook_base_path, get_modules

# Plan of attack: EN / ES / DE /  FR / SE
# 0. Translate the locale file
# 1. copy notebooks to notebooks/locale for each locale (delete old ones on demand)
# 2. For each set of localized notebooks, run translations for cell-sources in-place.
# 3. Generate course material for each locale.

API_KEY = os.environ.get("DEEPL_API_KEY")

# TODO we can work with glossaries to avoid translation issues, e.g. simply map
#  "policy" to "policy", don't use specific translations.
# https://www.deepl.com/docs-api/glossaries/create-glossary/


def translate_cell(text, locale):
    """Basic translation of a cell's source."""
    result = requests.get(
        "https://api-free.deepl.com/v2/translate",
        params={
            "auth_key": API_KEY,
            "target_lang": locale,
            "preserve_formatting": 1,
            "formality": "less",
            "text": text,
            "tag_handling": "html"
        },
    )
    return result.json()["translations"][0]["text"]


def translate_course_notebooks(locale):
    notebook_path = get_notebook_base_path()
    notebook_path = os.path.join(notebook_path, locale)
    modules = get_modules(notebook_path)

    for module in modules:

        readmes = [f.path for f in os.scandir(module) if f.is_file() if "README" in f.path]
        readmes.sort()
        notebooks = [f.path for f in os.scandir(module) if f.is_file() if ".ipynb" in f.path]
        notebooks.sort()

        for readme in readmes:
            with open(readme, "r") as f:
                content = f.read()
            content = translate_cell(content, locale)
            with open(readme, "w") as f:
                f.write(content)

        for nb_source in notebooks:

            print(f"Translating {nb_source} to {locale}")
            nb = get_notebook(nb_source)

            for cell in nb["cells"]:
                if cell["cell_type"] == "markdown":
                    cell["source"] = translate_cell(cell["source"], locale=locale)

                    # formatting issues
                    cell["source"] = cell["source"].replace("[ x]", "[x]")

                    # German "undos"
                    cell["source"] = cell["source"].replace("Verstärkungslernen", "Reinforcement Learning")
                    cell["source"] = cell["source"].replace("Umwelt", "Umgebung")
                    cell["source"] = cell["source"].replace("policy", "Policy")
                    cell["source"] = cell["source"].replace("Politik", "Policy")
                    cell["source"] = cell["source"].replace("politik", "policy")
                    cell["source"] = cell["source"].replace("Police", "Policy")
                    cell["source"] = cell["source"].replace("Strategie", "Policy")
                    cell["source"] = cell["source"].replace("Richtlinie", "Policy")
                    cell["source"] = cell["source"].replace("Wörterbuch", "dictionary")

            with open(nb_source, "w") as f:
                nbformat.write(nb, f)


if __name__ == "__main__":
    # TODO just automate this
    # Note: before running this, you need to set the DEEPL_API_KEY environment variable.
    # AND you need to copy all source notebooks into the respective locale sub-folders,
    # e.g. notebooks/00-introduction/00-introduction.ipynb
    #      --> notebooks/de/00-introduction/00-introduction.ipynb etc.
    translation_locales = []
    # translation_locales = ["de", "fr", "es"]
    for locale in translation_locales:
        translate_course_notebooks(locale=locale)
