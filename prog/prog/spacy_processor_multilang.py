import spacy
import subprocess
import sys
import json

MODEL_NAME = "xx_ent_wiki_sm"

def load_spacy_model():
    """
    Charge le modèle multilingue spaCy, ou le télécharge si nécessaire.
    """
    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        print(f"[INFO] Modèle spaCy '{MODEL_NAME}' non trouvé. Téléchargement...")
        subprocess.run([sys.executable, "-m", "spacy", "download", MODEL_NAME], check=True)
        return spacy.load(MODEL_NAME)

nlp = load_spacy_model()

def is_valid_token(token):
    """
    Définit si un token est pertinent pour les statistiques lexicales.
    """
    return (
        not token.is_punct and
        not token.is_space and
        not token.is_quote and
        not token.is_currency and
        token.text.strip() != ""
    )

def analyze_doc(doc):
    """
    Analyse un document spaCy pour extraire :
    - Lemmatisation + POS
    - Entités nommées
    - Statistiques : n_tokens, n_types, prop_lemmes, prop_propn
    """
    lemmes = []
    types_set = set()
    n_tokens = 0
    n_lemmes = 0
    n_propn = 0

    for token in doc:
        if is_valid_token(token):
            n_tokens += 1
            types_set.add(token.text.lower())

            if token.pos_ != "PROPN":
                lemma = token.lemma_.lower()
                pos = token.pos_
                lemmes.append(f"{lemma}_{pos}")
                n_lemmes += 1
            else:
                n_propn += 1

    return {
        "lemmes": lemmes,
        "entites": [(ent.text, ent.label_) for ent in doc.ents],
        "n_tokens": n_tokens,
        "n_types": len(types_set),
        "prop_lemmes": round(n_lemmes / n_tokens, 3) if n_tokens else 0.0,
        "prop_propn": round(n_propn / n_tokens, 3) if n_tokens else 0.0
    }

def process_texts_by_lang(corpus_by_lang):
    """
    Applique spaCy à chaque texte du corpus regroupé par langue.
    """
    result = {}

    for lang, texts in corpus_by_lang.items():
        print(f"\n[INFO] Traitement langue : {lang} ({len(texts)} textes)")
        lang_results = []
        total_lemmes = 0

        for i, text in enumerate(texts):
            doc = nlp(text)
            stats = analyze_doc(doc)
            lang_results.append(stats)
            total_lemmes += len(stats["lemmes"])

            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"  → {i + 1}/{len(texts)} textes traités")

        print(f"[INFO] Langue {lang} : {total_lemmes} lemmes extraits au total.")
        result[lang] = lang_results

    return result

def save_processed_data(result, output_path):
    """
    Sauvegarde le dictionnaire de résultats JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Résultats sauvegardés dans : {output_path}")
