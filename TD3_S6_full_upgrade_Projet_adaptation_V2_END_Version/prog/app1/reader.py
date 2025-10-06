import os
import glob
from bs4 import BeautifulSoup

def read_html_content(filepath):
    """
    Extrait et nettoie le texte d’un fichier HTML avec BeautifulSoup.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"[WARN] Impossible de lire {filepath} : {e}")
        return ""


def load_corpus_by_language(base_dir):
    """
    Parcourt tous les fichiers .html dans le corpus multilingue.
    Regroupe les textes par langue (nom du dossier parent).

    Retourne : dict[str, list[str]]
    """
    corpus = {}

    pattern = os.path.join(base_dir, "*", "**", "*.html")
    filepaths = glob.glob(pattern, recursive=True)

    for path in filepaths:
        parts = os.path.normpath(path).split(os.sep)

        # On cherche à capturer le dossier de langue (ex: fr, en, etc.)
        if len(parts) < 3:
            continue

        lang = parts[-3]  # ex: ../../corpus_multi/fr/sousdir/fichier.html → "fr"
        if lang not in corpus:
            corpus[lang] = []

        text = read_html_content(path)
        if text:
            corpus[lang].append(text)

    return corpus
