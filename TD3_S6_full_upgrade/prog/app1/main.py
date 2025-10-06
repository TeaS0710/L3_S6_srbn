import os
import glob
import numpy as np
from tqdm import tqdm  # Pour la barre de progression

# Imports depuis nos modules
from reader import parse_iob_file
from spacy_processor import lemmatize_tokens
from vectorizer import vectorize_tokens, compute_similarity
from clusterer import run_affinity_propagation, build_clusters_dict
from saver import save_result_for_file

"""
configuration des chemins de sortie et d'entrée
"""

DATA_DIR = "../../DATA"
RESULTS_DIR = "../../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Quels labels NER (IOB) conserver ? Ex. B-LOC, I-LOC
NER_PREFIXES = ["B-LOC", "I-LOC"]

# Utiliser le lemme + POS ?
USE_POS = True

# Programme Principal

def main():
    # Parcours de tous les .bio du répertoire
    pattern = os.path.join(DATA_DIR, "**", "*.bio")
    filepaths = glob.glob(pattern, recursive=True)
    if not filepaths:
        print("Aucun fichier .bio trouvé dans data Fin.")
        return

    # On utilise tqdm pour la progression
    for filepath in tqdm(filepaths, desc="Traitement des fichiers"):
        # Optionnel, si vous voulez un log plus verbeux

        # 1) Lecture de ce fichier (.bio)
        items = parse_iob_file(filepath)  # liste de (token, label)
        if not items:
            print(f"\n{filepath}: Fichier vide ou non lisible.")
            continue

        # 2) Filtrage NER
        #    On ne garde que les tokens dont le label figure dans NER_PREFIXES
        tokens_raw = []
        for (tok, lab) in items:
            if lab in NER_PREFIXES:
                tokens_raw.append(tok.lower())

        if not tokens_raw:
            print(f"\n{filepath}: Pas de tokens correspondants à ces labels. On passe au suivant.")
            continue

        # 3) Lemmatisation (et POS si USE_POS=True)
        tokens_lemma = lemmatize_tokens(tokens_raw, use_pos=USE_POS)

        # 4) Vectorisation (n-grammes de caractères)
        X, vect = vectorize_tokens(tokens_lemma, analyzer='char', ngram_range=(2,3))

        # 5) Similarité
        similarity_matrix = compute_similarity(X, metric='cosine')

        # 6) Clustering (AffinityPropagation)
        labels, centers_idx = run_affinity_propagation(similarity_matrix)
        n_clusters = len(np.unique(labels))
        #print(f"Nombre de clusters trouvés : {n_clusters}")

        # 7) Construction du dict de clusters
        clusters_dict = build_clusters_dict(labels, centers_idx, tokens_lemma)

        # 8) Sauvegarde dans un JSON individuel
        base_filename = os.path.basename(filepath)
        base_no_ext = os.path.splitext(base_filename)[0]
        out_json_name = f"clusters_{base_no_ext}.json"
        out_json_path = os.path.join(RESULTS_DIR, out_json_name)

        # Nous passons la liste des tokens (tokens_lemma) pour enregistrement
        save_result_for_file(
            filepath=filepath,
            similarity_matrix=similarity_matrix,
            clusters_dict=clusters_dict,
            out_json_path=out_json_path,
            used_tokens=tokens_lemma
        )

    print("\nTous les fichiers ont été traités.")


if __name__ == "__main__":
    main()
