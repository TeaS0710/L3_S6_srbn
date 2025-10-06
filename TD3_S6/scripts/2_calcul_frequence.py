#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import spacy
from collections import Counter

# --- Chemins d'entrée / sortie ---
INPUT_JSON = "../results/donnees_brutes.json"
OUTPUT_JSON = "../results/frequences.json"
RESULTS_DIR = "../results"

# Chargement d'un pipeline spaCy (fr_core_news_sm par exemple)
nlp = spacy.load("fr_core_news_sm")

def freq_dict(tokens):
    """
    Construit un Counter (ou dict simple) de fréquences pour une liste de tokens.
    """
    return dict(Counter(tokens))

def parse_bio_content(bio_text):
    """
    Parse le contenu d'un fichier .bio (chaîne de caractères).
    Retourne une liste de tokens (la 1ère colonne de chaque ligne).
    Format attendu d'une ligne :
        token   label

    Si une ligne est vide ou mal formée, on l'ignore.
    """
    tokens = []
    for line in bio_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # On suppose que parts = [token, label] ou plus
        if len(parts) >= 2:
            token = parts[0]
            tokens.append(token)
    return tokens

def compute_frequencies(data):
    """
    data est la structure lue du JSON (donnees_brutes.json), c-à-d:
    {
      "DAUDET": {
        "kraken": [
          {"filepath": "...", "content": "...(fichier bio)..."},
          ...
        ],
        "tesserac": [...],
        ...
      },
      "MAUPASSANT": { ... },
      ...
    }

    On retourne une structure de fréquences :
    {
      "DAUDET": {
        "kraken": [
          {
            "filepath": "...",
            "freq_split": {...},
            "freq_spacy": {...}
          },
          ...
        ],
        "tesserac": [...],
        ...
      },
      "MAUPASSANT": {...},
      ...
    }
    """

    results = {}
    for auteur, types_extractions in data.items():
        results[auteur] = {}
        for type_extraction, files_list in types_extractions.items():
            out_list = []
            for item in files_list:
                bio_text = item["content"]  # Contenu brut du .bio

                # 1) On parse le contenu .bio pour extraire la liste de tokens
                tokens_bio = parse_bio_content(bio_text)

                # 2) Approche "split" (basique)
                #    Ici, "split" n'a plus vraiment de sens si on a déjà tokenisé via .bio,
                #    mais on peut comparer la liste tokens_bio vs. un simple split sur espaces.
                #    => Soit on garde tokens_bio comme "split", soit on compare avec un raw split(bio_text).
                tokens_split = tokens_bio  # ou bien bio_text.split() si vous voulez tester
                freq_spl = freq_dict(tokens_split)

                # 3) Traitement via spaCy
                #    On redonne le texte reconstruit (ou le texte complet) à spaCy.
                #    Pour être logique, on peut ré-assembler tokens_bio en une phrase,
                #    ou alors on peut envoyer le contenu brut .bio.
                #    Ici, on envoie juste la liste de tokens_bio jointe par des espaces.
                doc = nlp(" ".join(tokens_bio))
                tokens_spacy = [tok.text for tok in doc]
                freq_spa = freq_dict(tokens_spacy)

                # 4) On stocke les fréquences calculées
                out_list.append({
                    "filepath": item["filepath"],
                    "freq_split": freq_spl,
                    "freq_spacy": freq_spa
                })

            results[auteur][type_extraction] = out_list
    return results

def main():
    # Création du répertoire de résultats si nécessaire
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Lecture du JSON produit par le script de collecte (donnees_brutes.json)
    with open(INPUT_JSON, "r", encoding="utf-8") as fin:
        data_collected = json.load(fin)

    # 2) Calcul des fréquences
    freq_data = compute_frequencies(data_collected)

    # 3) Sauvegarde au format JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(freq_data, fout, ensure_ascii=False, indent=4)
    print(f"Fichier de fréquences écrit : {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
