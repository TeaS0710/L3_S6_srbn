#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from bs4 import BeautifulSoup
import glob

def load_html_from_language_folder(lang_folder):
    """
    Parcourt récursivement un dossier de langue et extrait le texte de chaque fichier HTML.
    Retourne un dictionnaire {rel_path: texte}.
    """
    data = {}
    pattern = os.path.join(lang_folder, "**", "*.html")
    html_files = glob.glob(pattern, recursive=True)

    for full_path in html_files:
        rel_path = os.path.relpath(full_path, lang_folder)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator=" ")
                clean_text = " ".join(text.split())
                data[rel_path] = clean_text
        except Exception as e:
            print(f"[WARN] Problème fichier {full_path}: {e}")

    return data

def main():
    base_dir = "../data/corpus-multi"  # Modifier si besoin
    output_file = "../outputs/data_loaded.json"
    all_data = {}

    for lang in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang)
        if os.path.isdir(lang_path):
            print(f"[INFO] Traitement langue : {lang}")
            all_data[lang] = load_html_from_language_folder(lang_path)

    with open(output_file, "w", encoding="utf-8") as f_out:
        json.dump(all_data, f_out, ensure_ascii=False, indent=2)

    print(f"[OK] Données chargées dans {output_file}")

if __name__ == "__main__":
    main()
