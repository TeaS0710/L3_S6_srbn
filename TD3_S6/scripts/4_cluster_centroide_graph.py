#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Chemin vers le JSON d'entrée (données .bio concaténées)
INPUT_JSON = "../results/donnees_brutes.json"

# Chemins de sortie
RESULTS_DIR = "../results/results_clustering"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "clustering_pos_pca.png")
OUTPUT_CLUSTERS_JSON = os.path.join(RESULTS_DIR, "clusters_output.json")

def parse_bio_content(bio_text):
    """
    Parse un texte au format .bio (token POS ...).
    Retourne une liste (token, pos).
    """
    tokens_pos = []
    lines = bio_text.strip().split("\n")
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            token, pos = parts[0], parts[1]
            tokens_pos.append((token, pos))
    return tokens_pos

def main():
    # Lecture du JSON contenant les données au format .bio
    with open(INPUT_JSON, "r", encoding="utf-8") as fin:
        data_collected = json.load(fin)

    # Rassemble (token, pos, auteur, type_extraction) dans une liste
    all_items = []
    for auteur, dict_types in data_collected.items():
        for type_extraction, file_list in dict_types.items():
            for file_info in file_list:
                bio_text = file_info["content"]
                parsed_lines = parse_bio_content(bio_text)
                for (token, pos) in parsed_lines:
                    all_items.append((token, pos, auteur, type_extraction))

    # Filtrage par POS (ex. NOUN, PROPN)
    valid_pos_tags = {"NOUN", "PROPN"}
    filtered_items = [it for it in all_items if it[1] in valid_pos_tags]
    if not filtered_items:
        print("Aucun token après filtrage. Arrêt.")
        return

    # Prépare les différentes listes nécessaires
    tokens_list = [it[0] for it in filtered_items]
    pos_list = [it[1] for it in filtered_items]
    auteur_list = [it[2] for it in filtered_items]
    type_list = [it[3] for it in filtered_items]

    # Vectorisation (n-grammes de caractères)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3), stop_words=None)
    if not tokens_list:
        print("Liste de tokens vide, impossible de vectoriser.")
        return
    X = vectorizer.fit_transform(tokens_list)
    if X.shape[1] == 0:
        print("Vocabulaire vide après vectorisation. Arrêt.")
        return

    # Matrice de similarités pour AffinityPropagation
    dist_matrix = pairwise_distances(X, metric="cosine")
    similarity_matrix = -dist_matrix

    # Clustering
    clustering = AffinityPropagation(affinity="precomputed", damping=0.6, random_state=0)
    clustering.fit(similarity_matrix)
    labels = clustering.labels_
    centers = clustering.cluster_centers_indices_
    unique_labels = np.unique(labels)
    print(f"Nombre de clusters trouvés : {len(unique_labels)}")

    # Construction d'un dictionnaire de clusters
    clusters_dict = {}
    for cid in unique_labels:
        idx_in_cluster = np.where(labels == cid)[0]
        center_idx = centers[cid]
        centroid_token = tokens_list[center_idx]
        cluster_tokens = [tokens_list[i] for i in idx_in_cluster]
        cluster_pos = [pos_list[i] for i in idx_in_cluster]
        cluster_auteurs = [auteur_list[i] for i in idx_in_cluster]
        cluster_types = [type_list[i] for i in idx_in_cluster]
        clusters_dict[int(cid)] = {
            "centroid": centroid_token,
            "membres": cluster_tokens,
            "pos": cluster_pos,
            "auteurs": cluster_auteurs,
            "types": cluster_types
        }

    # Réduction de dimension (PCA) pour la visualisation
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X.toarray())

    # Plot
    plt.figure(figsize=(9, 6))
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20")

    for i, token in enumerate(tokens_list):
        cid = labels[i]
        color = cmap(cid % 20)
        x, y = X_2d[i, 0], X_2d[i, 1]
        if i in centers:
            plt.scatter(x, y, s=120, c=[color], edgecolor='k')
            annot = f"{token} ({auteur_list[i]}, {type_list[i]})"
            plt.text(x+0.01, y+0.01, annot, fontsize=8)
        else:
            plt.scatter(x, y, s=30, c=[color])

    plt.title("Clustering AffinityPropagation - POS Filtrés")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.show()
    print(f"Graphique enregistré dans : {OUTPUT_PLOT}")

    # Sauvegarde en JSON
    with open(OUTPUT_CLUSTERS_JSON, "w", encoding="utf-8") as fout:
        json.dump(clusters_dict, fout, ensure_ascii=False, indent=4)
    print(f"Clusters sauvegardés dans : {OUTPUT_CLUSTERS_JSON}")

if __name__ == "__main__":
    main()
