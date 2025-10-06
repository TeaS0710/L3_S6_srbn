import numpy as np
from sklearn.cluster import AffinityPropagation
import time

#Exécuter AffinityPropagation avec plus de logs
#Retourner à (labels, cluster_centers_indices)
def run_affinity_propagation(similarity_matrix, random_state=42):
    print("[CLUSTER] → Initialisation de l'algorithme AffinityPropagation")

    n_points = similarity_matrix.shape[0]
    preference_val = np.median(similarity_matrix[np.nonzero(similarity_matrix)])
    damping_val = 0.65

    print(f"[CLUSTER] → Nombre de points à clusteriser : {n_points}")
    print(f"[CLUSTER] → Préférence (valeur médiane) : {preference_val:.4f}")
    print(f"[CLUSTER] → Paramètres : damping={damping_val}, max_iter=1000, convergence_iter=15")

    ap = AffinityPropagation(
        affinity='precomputed',
        random_state=random_state,
        max_iter=1000,
        convergence_iter=15,
        damping=damping_val,
        preference=preference_val
    )

    start = time.time()
    ap.fit(similarity_matrix)
    end = time.time()

    n_clusters = len(np.unique(ap.labels_))

    print(f"[CLUSTER] ✅ Clustering terminé en {end - start:.2f} sec")
    print(f"[CLUSTER] → Nombre de clusters trouvés : {n_clusters}")

    return ap.labels_, ap.cluster_centers_indices_

#Construire un dictionnaire de clusters avec centroid + membres
def build_clusters_dict(labels, centers_idx, tokens):
    print(f"[CLUSTER] → Construction du dictionnaire de clusters...")

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

    print(f"[CLUSTER] ✅ Dictionnaire construit : {len(clusters)} clusters enregistrés.")
    return clusters
