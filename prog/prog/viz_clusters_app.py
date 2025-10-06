import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import MDS
import matplotlib.cm as cm

def build_labels_from_clusters(tokens, clusters):
    token_to_cid = {}
    cluster_sizes = {}
    centroids_idx = {}

    for cid_str, cdata in clusters.items():
        cid = int(cid_str)
        members = cdata["members"]
        cluster_sizes[cid] = len(members)
        centroid_token = cdata["centroid"]

        for mem in members:
            token_to_cid[mem] = cid

        if centroid_token in tokens:
            idx_centroid = tokens.index(centroid_token)
            centroids_idx[cid] = idx_centroid
        else:
            centroids_idx[cid] = -1

    labels = []
    for t in tokens:
        cid = token_to_cid.get(t, -1)
        labels.append(cid)

    return np.array(labels), cluster_sizes, centroids_idx

def parse_pos_from_token(token):
    if "_" in token:
        return token.split("_")[-1]
    return None

def pos_to_marker(pos):
    marker_map = {
        "NOUN": "o",
        "PROPN": "D",
        "VERB": "s",
        "ADJ": "^",
        "ADV": "v"
    }
    return marker_map.get(pos, 'x')

def plot_mds_2d(ax, tokens, sim_mat, clusters, title):
    plt.style.use('seaborn-v0_8')
    labels, cluster_sizes, centroids_idx = build_labels_from_clusters(tokens, clusters)
    dist_mat = 1.0 - sim_mat

    mds_2d = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_2d = mds_2d.fit_transform(dist_mat)

    cmap = cm.get_cmap("tab20")

    for i, token in enumerate(tokens):
        cid = labels[i]
        color = "gray" if cid < 0 else cmap(cid % 20)
        pos_tag = parse_pos_from_token(token)
        marker = pos_to_marker(pos_tag)
        x, y = pos_2d[i, 0], pos_2d[i, 1]

        is_centroid = any(idx == i for idx in centroids_idx.values())
        if is_centroid:
            c_found = [c for c, c_idx in centroids_idx.items() if c_idx == i]
            if c_found:
                c_id = c_found[0]
                size = 20 + cluster_sizes[c_id] * 2
            else:
                size = 30
            if marker == 'x':
                ax.scatter(x, y, c=[color], marker=marker, s=size, alpha=0.8, linewidth=0.7)
            else:
                ax.scatter(x, y, c=[color], marker=marker, s=size, edgecolors='black', alpha=0.8, linewidth=0.7)
        else:
            if marker == 'x':
                ax.scatter(x, y, c=[color], marker=marker, s=15, alpha=0.7, linewidth=0.7)
            else:
                ax.scatter(x, y, c=[color], marker=marker, s=15, alpha=0.7, edgecolors='none')

    ax.set_title(title, fontsize=10)
    ax.grid(True)

def load_data_for_lang(lang, processed_path, clusters_path):
    with open(processed_path, "r", encoding="utf-8") as f:
        processed = json.load(f)
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    if lang not in processed or lang not in clusters:
        raise ValueError(f"Données manquantes pour la langue : {lang}")

    all_lemmes = []
    for doc in processed[lang]:
        all_lemmes.extend(doc["lemmes"])

    cluster_tokens = sorted(set(tok for c in clusters[lang].values() for tok in c["members"]))
    tokens = [t for t in all_lemmes if t in cluster_tokens]

    return tokens, clusters[lang]

def vectorize(tokens, ngram_range=(2, 3)):
    vect = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    X = vect.fit_transform(tokens)
    return X

def build_similarity(X):
    return 1.0 - pairwise_distances(X, metric="cosine")

def visualize_clusters(lang, processed_path, clusters_path, ngram_range=(2, 3)):
    print(f"[INFO] Visualisation MDS pour la langue : {lang}")

    tokens, clusters = load_data_for_lang(lang, processed_path, clusters_path)
    if len(tokens) < 3 or len(clusters) < 1:
        print(f"[WARN] Trop peu de données pour {lang}")
        return

    X = vectorize(tokens, ngram_range)
    sim_matrix = build_similarity(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_mds_2d(ax, tokens, sim_matrix, clusters, f"MDS des clusters – {lang}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    LANG = "fr"  # Change cette valeur pour visualiser d'autres langues
    BASE_DIR = "../pipeline_results"
    processed_path = os.path.join(BASE_DIR, "processed_multilang.json")
    clusters_path = os.path.join(BASE_DIR, "clusters_ngrams_2_3.json")

    visualize_clusters(LANG, processed_path, clusters_path, ngram_range=(2, 3))
