import os
import glob
import json

DATA_DIR = "../DATA"          # Répertoire racine contenant {AUTEUR}/{TYPE}/*.bio
RESULTS_DIR = "../results"    # Répertoire de sortie
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(RESULTS_DIR, "donnees_brutes.json")

def main():
    # Patron de recherche : on parcourt tous les fichiers .bio
    pattern = os.path.join(DATA_DIR, "**/*.bio")

    results = {}

    # glob.glob(..., recursive=True) pour parcourir toute l'arborescence
    for filepath in glob.glob(pattern, recursive=True):
        # Exemple de filepath :
        #   DATA/DAUDET/kraken/mon_fichier.bio
        # On va splitter sur os.sep :
        splitted = filepath.split(os.sep)

        # splitted[-1] => "mon_fichier.bio"
        # splitted[-2] => "kraken" (type d'extraction)
        # splitted[-3] => "DAUDET" (auteur)
        # (si votre structure est plus profonde, adaptez les indices)

        auteur = splitted[-3]
        type_extraction = splitted[-2]

        with open(filepath, "r", encoding="utf-8") as fin:
            content = fin.read()

        # On range le contenu dans une structure de type :
        # results = {
        #   "DAUDET": {
        #       "kraken": [
        #           {"filepath": "...", "content": "..."},
        #           ...
        #       ],
        #       "tesserac": [...],
        #       ...
        #   },
        #   "MAUPASSANT": {...},
        #   ...
        # }

        if auteur not in results:
            results[auteur] = {}
        if type_extraction not in results[auteur]:
            results[auteur][type_extraction] = []

        results[auteur][type_extraction].append({
            "filepath": filepath,
            "content": content
        })

    # Écriture du JSON final
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=4)
    print(f"Fichier JSON écrit : {OUTPUT_JSON}")

if __name__ == "__1main__":
    main()
