import json
import glob
import re
from collections import Counter, OrderedDict

def read_iob2_file(filepath):
    """
    Lit un fichier au format IOB2 et renvoie
    un dictionnaire ou une liste contenant les tokens d'entités nommées.
    """
    named_entities = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Chaque ligne contient: token label
            parts = line.split()
            if len(parts) == 2:
                token, label = parts
                if label != "O":  # On ne garde que les entités nommées
                    named_entities.append(token)
            # Gestion d'éventuelles erreurs de format
            # else:
            #     ...

    return named_entities

def collect_named_entities_from_iob2(directory_path):
    """
    Parcourt tous les fichiers .iob2 dans un répertoire et
    retourne l'ensemble des tokens d'entités nommées.
    """
    all_entities = []
    for file in glob.glob(directory_path + "/*.iob2"):
        tokens = read_iob2_file(file)
        all_entities.extend(tokens)
    return all_entities

def clean_token(token):
    """
    Normalise le token : on peut décider d'abaisser la casse,
    d'enlever certains caractères spéciaux, etc.
    """
    token = token.strip()
    # Exemple : suppression des caractères non alphanumériques en début/fin
    token = re.sub(r"^\W+|\W+$", "", token)
    # Passage en minuscule
    token = token.lower()
    return token

def normalize_tokens(tokens):
    """
    Applique le nettoyage à une liste de tokens
    et retourne la liste nettoyée.
    """
    return [clean_token(t) for t in tokens if t.strip() != ""]

def aggregate_tokens(tokens):
    """
    Retourne un Counter contenant token -> fréquence.
    """
    return Counter(tokens)

def get_unique_tokens(token_counter):
    """
    Retourne la liste unique des tokens (clés du Counter).
    """
    return list(token_counter.keys())

import numpy as np
import sklearn
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer

def cluster_tokens_with_affprop(tokens, token_counter, ngram_range=(2,3)):
    """
    Regroupe en clusters les tokens fournis en entrée.
    Utilise la vectorisation n-grammes (2,3) et la distance cosinus,
    puis l'algorithme d'AffinityPropagation.

    :param tokens: liste de tokens uniques
    :param token_counter: Counter(token -> fréquence)
    :param ngram_range: tuple (min_n, max_n) pour le CountVectorizer
    :return: un dictionnaire décrivant chaque cluster
    """
    # Préparation de la matrice de distance
    words = np.asarray(tokens)
    size = len(words)
    # Construction de la matrice de similarités
    # (attention: on va construire la matrice "distance", puis on la rend négative
    # pour l'utiliser avec AffinityPropagation)
    matrice = []

    for w in words:
        vect_row = []
        for w2 in words:
            V = CountVectorizer(ngram_range=ngram_range, analyzer='char')
            X = V.fit_transform([w, w2]).toarray()
            dist_cos = sklearn.metrics.pairwise.cosine_distances(X)[0][1]
            vect_row.append(dist_cos)
        matrice.append(vect_row)

    matrice_np = np.array(matrice)
    # Pour l'AffinityPropagation, il faut fournir une matrice de similarités
    # => On prend l'opposé de la distance (distance * -1)
    matrice_def = -1 * matrice_np

    # Application de l'affinity propagation
    affprop = AffinityPropagation(
        affinity="precomputed",
        damping=0.6,
        random_state=None
    )
    affprop.fit(matrice_def)

    # Construction du résultat
    clusters = {}
    i = 1
    for cluster_id in np.unique(affprop.labels_):
        # Centroïde (exemplar)
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster_members = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])

        # On prépare la structure de sortie
        cluster_key = f"ID {i}"
        clusters[cluster_key] = {
            "Centroïde": exemplar,
            "Freq. centroïde": token_counter[exemplar],
            "Termes": list(cluster_members),
        }
        i += 1

    return clusters

import json
import os

def main_iob2_clustering(input_dir, output_json):
    # 1. Récupération des tokens
    all_entities = collect_named_entities_from_iob2(input_dir)

    # 2. Nettoyage
    clean_entities = normalize_tokens(all_entities)

    # 3. Agrégation (fréquences)
    token_counter = aggregate_tokens(clean_entities)

    # 4. Liste unique des tokens
    unique_tokens = get_unique_tokens(token_counter)

    # 5. Clustering
    clusters = cluster_tokens_with_affprop(unique_tokens, token_counter, ngram_range=(2,3))

    # 6. Sauvegarde JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=4, ensure_ascii=False)

    print(f"Clustering terminé. Résultat disponible dans : {output_json}")


if __name__ == "__main__":
    # Exemple d'utilisation
    input_directory = "chemin_vers_dossier_iob2"
    output_file = "clusters_output.json"

    # On lance le pipeline
    main_iob2_clustering(input_directory, output_file)
