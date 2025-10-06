#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import matplotlib.pyplot as plt

INPUT_FREQ_JSON = "../results/frequences.json"
RESULTS_DIR = "../results"

def zipf_plot(freq_s, freq_sp, title_suffix):
    """
    freq_s : dict des fréquences calculées avec split
    freq_sp: dict des fréquences calculées avec spacy
    title_suffix : texte à ajouter dans le titre (ex: "DAUDET / kraken")
    """
    # Si aucune fréquence, on sort
    if not freq_s or not freq_sp:
        return

    vs = sorted(freq_s.values(), reverse=True)
    vp = sorted(freq_sp.values(), reverse=True)

    plt.figure(figsize=(6, 4))
    plt.loglog(range(1, len(vs) + 1), vs, label='split')
    plt.loglog(range(1, len(vp) + 1), vp, label='spaCy')
    plt.title(f"Loi de Zipf - {title_suffix}")
    plt.xlabel('Rang')
    plt.ylabel('Fréquence')
    plt.legend()

    # On nettoie le titre pour en faire un nom de fichier plus sûr
    filename_suffix = title_suffix.replace(" ", "_").replace("/", "_")
    out_png = os.path.join(RESULTS_DIR, f'zipf_{filename_suffix}.png')
    plt.savefig(out_png)
    plt.show()
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Lecture du JSON produit par le script 2
    with open(INPUT_FREQ_JSON, "r", encoding="utf-8") as fin:
        freq_data = json.load(fin)

    # freq_data a la forme:
    # {
    #   "DAUDET": {
    #       "kraken": [
    #         {
    #           "filepath": "...",
    #           "freq_split": {...},
    #           "freq_spacy": {...}
    #         },
    #         ...
    #       ],
    #       "tesserac": [...],
    #       ...
    #   },
    #   "MAUPASSANT": { ... },
    #   ...
    # }

    for auteur, types_extractions in freq_data.items():
        for type_extraction, files_list in types_extractions.items():
            # On peut générer un plot par fichier, ou un plot agrégé par type...
            # Exemple : un plot par fichier
            for d in files_list:
                # Récupération des dict de fréquences
                freq_spl = d["freq_split"]
                freq_spa = d["freq_spacy"]

                # Création d'un suffix pour le titre
                suffix = f"{auteur} / {type_extraction} / {os.path.basename(d['filepath'])}"
                zipf_plot(freq_spl, freq_spa, suffix)

    print("Plots de Zipf générés dans le dossier results.")

if __name__ == "__main__":
    main()
