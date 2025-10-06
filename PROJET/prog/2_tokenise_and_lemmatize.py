import json
import spacy

def load_lang_model(lang_code):
    """
    Renvoie le modèle spaCy adapté à la langue.
    Par ex. : fr -> fr_core_news_md, en -> en_core_web_md, ...
    """
    if lang_code == "fr":
        return spacy.load("fr_core_news_md")
    elif lang_code == "en":
        return spacy.load("en_core_web_md")
    # Ajouter d’autres langues si besoin ...
    else:
        raise ValueError(f"Langue non gérée : {lang_code}")

def main():
    input_file = "../outputs/data_loaded.json"
    output_file = "../outputs/data_lemmatized.json"

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    result = {}

    for lang_code, texts_dict in all_data.items():
        nlp = load_lang_model(lang_code)
        result[lang_code] = {}
        for file_name, text_content in texts_dict.items():
            doc = nlp(text_content)
            tokens_and_lemmas = [(token.text, token.lemma_) for token in doc]
            result[lang_code][file_name] = tokens_and_lemmas

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
