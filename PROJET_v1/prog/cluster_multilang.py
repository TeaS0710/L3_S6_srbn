import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation

def vectorize_tokens(tokens, analyzer='char', ngram_range=(2, 3)):
    """
    Vectorise la liste de tokens en utilisant CountVectorizer
    (par défaut, n-grammes de caractères).
    """
    vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokens)
    return X, vectorizer

def compute_similarity(X, metric='cosine'):
    """
    Calcule la matrice de similarité à partir de la matrice X.
    (distance cosinus => similarité = 1 - distance)
    """
    dist_matrix = pairwise_distances(X, metric=metric)
    similarity = 1.0 - dist_matrix
    return similarity

def run_affinity_propagation(similarity_matrix, random_state=42):
    """
    Exécute l'algo AffinityPropagation sur la matrice de similarité.
    Retourne (labels, cluster_centers_indices).
    """
    ap = AffinityPropagation(
        affinity='precomputed',
        random_state=random_state,
        max_iter=1000,
        damping=0.7,
        preference=np.median(similarity_matrix),
        convergence_iter=15
    )
    ap.fit(similarity_matrix)
    return ap.labels_, ap.cluster_centers_indices_

def build_clusters_dict(labels, centers_idx, tokens):
    """
    Construit un dict décrivant chaque cluster :
    - 'centroid' = token représentant
    - 'members' = liste de tokens
    """
    clusters = {}
    unique_labels = np.unique(labels)

    for cid in unique_labels:
        idx_in_cluster = np.where(labels == cid)[0]
        center_idx = centers_idx[cid]

        centroid_token = tokens[center_idx]
        members = [tokens[i] for i in idx_in_cluster]

        clusters[int(cid)] = {
            "centroid": centroid_token,
            "members": members
        }

    return clusters

def cluster_all_languages(input_path, output_path, ngram_range=(2, 3)):
    """
    Pour chaque langue, applique un clustering sur les lemmes.
    Stocke les résultats dans un fichier JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for lang, docs in data.items():
        print(f"[INFO] Clustering langue : {lang} avec ngrammes {ngram_range}")

        # Rassembler tous les lemmes de cette langue
        all_lemmes = []
        for doc in docs:
            all_lemmes.extend(doc.get("lemmes", []))

        # Nettoyage : dédoublonner + ignorer les très courts
        tokens = sorted(set([t for t in all_lemmes if len(t) >= 3]))
        if len(tokens) < 5:
            print(f"[WARN] Pas assez de lemmes pour clusteriser {lang}.")
            continue

        # Vectorisation
        X, _ = vectorize_tokens(tokens, analyzer="char", ngram_range=ngram_range)

        # Similarité
        similarity_matrix = compute_similarity(X)

        # Clustering
        labels, centers_idx = run_affinity_propagation(similarity_matrix)

        # Dictionnaire de clusters
        clusters = build_clusters_dict(labels, centers_idx, tokens)

        results[lang] = clusters
        print(f"→ {lang} : {len(clusters)} clusters trouvés.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Clusters sauvegardés dans : {output_path}")
