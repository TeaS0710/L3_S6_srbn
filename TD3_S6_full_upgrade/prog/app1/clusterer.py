import numpy as np
from sklearn.cluster import AffinityPropagation

def run_affinity_propagation(similarity_matrix, random_state=42):
    """
    Exécute l'algo AffinityPropagation sur la matrice de similarité.
    Retourne (labels, cluster_centers_indices).
    """
    ap = AffinityPropagation(
        affinity='precomputed',
        random_state=42,
        max_iter=1000,          # Increase from 200 to 1000
        damping=0.7,   # Increase the damping
        preference=np.median(similarity_matrix),  # or some other heuristic
        convergence_iter=10,    # Check convergence every 50 iterations
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
