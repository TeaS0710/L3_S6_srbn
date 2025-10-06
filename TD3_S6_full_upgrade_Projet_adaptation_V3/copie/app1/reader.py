
import os
import glob
from bs4 import BeautifulSoup

#Extraire et nettoyer le texte d’un fichier HTML avec BeautifulSoup
def read_html_content(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"[WARN] Impossible de lire {filepath} : {e}")
        return ""

#Parcourir tous les fichiers .html dans le corpus multilingue
#Regrouper les textes par langue
#Retourner à dict[str, list[str]]
def load_corpus_by_language(base_dir):
    corpus = {}

    pattern = os.path.join(base_dir, "*", "**", "*.html")
    filepaths = glob.glob(pattern, recursive=True)

    for path in filepaths:
        parts = os.path.normpath(path).split(os.sep)

        # On cherche à capturer le dossier de langue
        if len(parts) < 3:
            continue

        lang = parts[-3]  #récupérer le nom raccourci de la langue à partir du chemin
        if lang not in corpus:
            corpus[lang] = []

        text = read_html_content(path)
        if text:
            corpus[lang].append(text)

    return corpus
