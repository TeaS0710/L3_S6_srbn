import json
import spacy

def load_lang_model(lang_code):
    # Même logique qu'avant
    if lang_code == "fr":
        return spacy.load("fr_core_news_md")
    elif lang_code == "en":
        return spacy.load("en_core_web_md")
    else:
        raise ValueError(f"Langue non gérée : {lang_code}")

def main():
    input_file = "../outputs/data_lemmatized.json"
    output_file = "../outputs/data_after_ner.json"

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    cleaned_data = {}
    entities_info = {}

    for lang_code, texts_dict in all_data.items():
        nlp = load_lang_model(lang_code)
        cleaned_data[lang_code] = {}
        entities_info[lang_code] = {}

        for file_name, tokens_and_lemmas in texts_dict.items():
            # Reconstruire le texte brut
            text_rebuilt = " ".join([tok for tok, lem in tokens_and_lemmas])
            doc = nlp(text_rebuilt)

            # Récupérer entités nommées
            named_entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities_info[lang_code][file_name] = named_entities

            # Filtrage
            filtered_lemmas = []
            for token in doc:
                if (not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and token.pos_ != "PROPN"):
                    # On garde le lemma en minuscules
                    filtered_lemmas.append(token.lemma_.lower())

            cleaned_data[lang_code][file_name] = filtered_lemmas

    # Sauvegarde
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    ner_output_file = "../outputs/named_entities.json"
    with open(ner_output_file, "w", encoding="utf-8") as f:
        json.dump(entities_info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
