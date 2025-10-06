import spacy
import subprocess
import sys
import json

MODEL_NAME = "xx_ent_wiki_sm"

def load_spacy_model():
    """
    Charge le modèle multilingue de spaCy, en le téléchargeant si nécessaire.
    """
    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        print(f"[INFO] Modèle spaCy '{MODEL_NAME}' non trouvé. Téléchargement en cours...")
        subprocess.run([sys.executable, "-m", "spacy", "download", MODEL_NAME], check=True)
        return spacy.load(MODEL_NAME)

nlp = load_spacy_model()

def analyze_doc(doc):
    """
    Extrait les informations d'un document spaCy :
    - Lemmatisation
    - POS
    - NER
    - Statistiques lexicales
    """
    lemmes = []
    n_tokens = 0
    n_types = set()
    n_lemmes = 0
    n_propn = 0

    for token in doc:
        if token.is_alpha and not token.is_stop:
            n_tokens += 1
            n_types.add(token.text.lower())

            if token.pos_ != "PROPN":
                lemma = token.lemma_.lower()
                pos = token.pos_
                lemmes.append(f"{lemma}_{pos}")
                n_lemmes += 1
            else:
                n_propn += 1

    entites = [(ent.text, ent.label_) for ent in doc.ents]

    stats = {
        "lemmes": lemmes,
        "entites": entites,
        "n_tokens": n_tokens,
        "n_types": len(n_types),
        "prop_lemmes": round(n_lemmes / n_tokens, 3) if n_tokens else 0.0,
        "prop_propn": round(n_propn / n_tokens, 3) if n_tokens else 0.0
    }

    return stats

def process_texts_by_lang(corpus_by_lang):
    """
    Applique l'analyse linguistique spaCy à chaque texte du corpus par langue.
    """
    result = {}

    for lang, texts in corpus_by_lang.items():
        print(f"\n[INFO] Traitement langue : {lang} ({len(texts)} textes)")
        lang_results = []

        for i, text in enumerate(texts):
            doc = nlp(text)
            doc_stats = analyze_doc(doc)
            lang_results.append(doc_stats)

            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"  → {i + 1}/{len(texts)} textes traités")

        result[lang] = lang_results

    return result

def save_processed_data(result, output_path):
    """
    Sauvegarde les résultats au format JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Résultats sauvegardés dans : {output_path}")
