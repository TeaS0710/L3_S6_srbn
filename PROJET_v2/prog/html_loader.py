import os
import glob
import json
from bs4 import BeautifulSoup

def extract_text_from_html(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"[ERREUR] {filepath} : {e}")
        return None

def load_corpus_by_language(base_dir):
    """
    Charge tous les fichiers HTML pour chaque langue dans base_dir/langue/**.html
    Regroupe les textes par langue dans un dictionnaire {langue: [texte1, texte2, ...]}
    """
    corpus = {}

    # Les sous-dossiers immédiats sont considérés comme des codes de langue
    for lang in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang)
        if not os.path.isdir(lang_path):
            continue

        # Cherche tous les .html dans tous les sous-dossiers
        pattern = os.path.join(lang_path, "**", "*.html")
        filepaths = glob.glob(pattern, recursive=True)

        texts = []
        for path in filepaths:
            text = extract_text_from_html(path)
            if text:
                texts.append(text)

        if texts:
            corpus[lang] = texts
            print(f"[INFO] {lang} : {len(texts)} fichiers chargés.")
        else:
            print(f"[WARN] Aucun texte valide pour la langue : {lang}")

    return corpus

def save_corpus_to_json(corpus_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_dict, f, ensure_ascii=False, indent=2)
    print(f"[OK] Corpus sauvegardé dans : {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    BASE_DIR = "../../corpus-multi"
    OUTPUT_JSON = "../../corpus_grouped_by_lang.json"

    corpus = load_corpus_by_language(BASE_DIR)
    save_corpus_to_json(corpus, OUTPUT_JSON)
