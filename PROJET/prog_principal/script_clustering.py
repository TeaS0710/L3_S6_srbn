# script_clustering.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def load_all_tokens(json_files):
    all_tokens = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            all_tokens.extend(doc["tokens"])
    return all_tokens

def cluster_tokens(all_tokens, ngram_range, out_prefix):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    X = vectorizer.fit_transform(all_tokens)
    dist_mat = pairwise_distances(X, metric='cosine')
    sim_mat = 1.0 - dist_mat
    ap = AffinityPropagation(affinity='precomputed', random_state=42, preference=np.median(sim_mat))
    ap.fit(sim_mat)
    labels = ap.labels_
    centers_idx = ap.cluster_centers_indices_

    clusters = {}
    for cid in np.unique(labels):
        idxs = np.where(labels == cid)[0]
        clusters[int(cid)] = {
            "centroid": all_tokens[centers_idx[cid]],
            "members": [all_tokens[i] for i in idxs]
        }

    out_json = f"{out_prefix}_clusters.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_clusters": len(clusters), "clusters": clusters}, f, ensure_ascii=False, indent=2)

    coords_2d = PCA(n_components=2).fit_transform(X.toarray())
    plt.figure()
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        plt.scatter(coords_2d[idx,0], coords_2d[idx,1], label=f"Cluster {cid}")
    plt.title(f"Clustering n-gram={ngram_range}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pca.png")
    plt.close()
