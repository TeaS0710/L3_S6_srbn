#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.cm as cm
from collections import Counter

# ========== PARAMÈTRES PAR DÉFAUT ==========

DEFAULT_JSON_FOLDER = "../../results"
DEFAULT_OUTPUT_FOLDER = "../../results"

# ========== UTILS DE CHARGEMENT ==========

def collect_json_files(json_folder):
    return glob.glob(os.path.join(json_folder, "*.json"))

def load_clusters_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = data["tokens"]
    sim_mat = np.array(data["similarity_matrix"])
    clusters = data["clusters"]
    file_origin = data.get("file", os.path.basename(json_path))
    return tokens, sim_mat, clusters, file_origin

# ========== STATISTIQUES CLUSTERS ==========

def print_cluster_stats(json_dir):
    print("\n========== STATISTIQUES CLUSTERING ==========")
    json_files = collect_json_files(json_dir)
    for json_path in json_files:
        tokens, sim_mat, clusters, file_origin = load_clusters_json(json_path)
        n_tokens = len(tokens)
        n_clusters = len(clusters)
        dist_mat = 1 - sim_mat
        avg_distance = np.mean(dist_mat[np.triu_indices(n_tokens, k=1)]) if n_tokens > 1 else 0.0
        cluster_sizes = [len(c["members"]) for c in clusters.values()]
        inter_size = sum(len(set(c1["members"]).intersection(set(c2["members"])))
                         for i, c1 in clusters.items() for j, c2 in clusters.items() if i < j)

        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        min_cluster_size = min(cluster_sizes) if cluster_sizes else 0
        singleton_clusters = sum(1 for s in cluster_sizes if s == 1)
        tokens_in_clusters = sum(cluster_sizes)
        coverage_ratio = tokens_in_clusters / n_tokens if n_tokens else 0

        print(f"\n→ {file_origin}")
        print(f"  Tokens totaux         : {n_tokens}")
        print(f"  Tokens clusterisés    : {tokens_in_clusters} ({coverage_ratio:.1%})")
        print(f"  Clusters              : {n_clusters}")
        print(f"  Clusters singletons   : {singleton_clusters}")
        print(f"  Moy. distance globale : {avg_distance:.4f}")
        print(f"  Intersections totales : {inter_size}")
        print(f"  Taille moy. cluster   : {avg_cluster_size:.2f}")
        print(f"  Taille min / max      : {min_cluster_size} / {max_cluster_size}")

        if n_clusters > 1:
            purity = max(cluster_sizes) / n_tokens
            print(f"  Purity approx         : {purity:.3f}")

# ========== MDS CLUSTER VISUALIZATION ==========

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
        centroids_idx[cid] = tokens.index(centroid_token) if centroid_token in tokens else -1
    labels = [token_to_cid.get(t, -1) for t in tokens]
    return np.array(labels), cluster_sizes, centroids_idx

def parse_pos_from_token(token):
    return token.split("_")[-1] if "_" in token else None

def pos_to_marker(pos):
    return {
        "NOUN": "o", "PROPN": "D", "VERB": "s", "ADJ": "^", "ADV": "v"
    }.get(pos, 'x')

def plot_mds_2d(ax, tokens, sim_mat, clusters, title):
    plt.style.use('seaborn-v0_8')
    labels, cluster_sizes, centroids_idx = build_labels_from_clusters(tokens, clusters)
    dist_mat = 1.0 - sim_mat
    pos_2d = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_mat)
    cmap = cm.get_cmap("tab10")

    top_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])[:5]
    top_cluster_ids = set(cid for cid, _ in top_clusters)

    for i, token in enumerate(tokens):
        cid = labels[i]
        color = cmap(cid % 10) if cid in top_cluster_ids else 'lightgray'
        pos_tag = parse_pos_from_token(token)
        marker = pos_to_marker(pos_tag)
        x, y = pos_2d[i, 0], pos_2d[i, 1]
        is_centroid = any(idx == i for idx in centroids_idx.values())
        size = 40 if is_centroid else 15
        kwargs = dict(c=[color], marker=marker, s=size, alpha=0.9 if is_centroid else 0.5)
        if marker != 'x':
            kwargs["edgecolors"] = 'black' if is_centroid else 'none'
        ax.scatter(x, y, **kwargs)

    legend_labels = [f"Cluster {cid} ({cluster_sizes[cid]} mots)" for cid in top_cluster_ids]
    for i, label in enumerate(legend_labels):
        ax.scatter([], [], c=[cmap(i)], label=label)

    ax.set_title(title + "\n(Représentation des 5 plus gros clusters par couleur)", fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc='best', fontsize=8, frameon=True)
    ax.grid(True)

# ========== N-GRAM PARTITIONING ==========

def plot_partition(words, ngram_range, title, out_path):
    if not words:
        print(f"[WARN] Aucun mot à afficher pour : {title}")
        return
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range)
    X = vectorizer.fit_transform(words)
    if X.shape[0] < 2:
        print(f"[WARN] Trop peu de données pour {title} (n={X.shape[0]})")
        return
    dist = cosine_distances(X.toarray())
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(dist)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.7, c='dodgerblue')
    ax.set_title(title + "\n(Projection MDS sur les similarités de ngrammes)", fontsize=11)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def generate_ngram_partitionings(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for json_path in collect_json_files(json_dir):
        tokens, _, _, file_origin = load_clusters_json(json_path)
        if not tokens:
            print(f"[WARN] {json_path} : aucun token trouvé.")
            continue
        base = os.path.basename(json_path).replace(".json", "")
        for nrange in [(2, 3), (4, 5)]:
            title = f"{file_origin} - ngrammes {nrange}"
            fname = f"{base}_ngrams_{nrange[0]}_{nrange[1]}.png"
            out_path = os.path.join(output_dir, fname)
            print(f"[PARTITION] {file_origin} → {fname}")
            plot_partition(tokens, ngram_range=nrange, title=title, out_path=out_path)

# ========== MAIN ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse et visualisation des clusters lexicaux.")
    parser.add_argument("json_dir", nargs='?', default=DEFAULT_JSON_FOLDER,
                        help="Dossier contenant les fichiers JSON (par défaut : %(default)s).")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_FOLDER,
                        help="Répertoire de sortie pour les images (par défaut : %(default)s).")
    args = parser.parse_args()

    print_cluster_stats(args.json_dir)
    generate_ngram_partitionings(args.json_dir, args.outdir)

    json_files = collect_json_files(args.json_dir)
    os.makedirs(args.outdir, exist_ok=True)
    for json_path in json_files:
        tokens, sim_mat, clusters, file_origin = load_clusters_json(json_path)
        fig, ax = plt.subplots(figsize=(10, 7))
        title = f"MDS - {file_origin}"
        plot_mds_2d(ax, tokens, sim_mat, clusters, title)
        fig.tight_layout()
        fname = os.path.splitext(os.path.basename(json_path))[0] + "_mds.png"
        fig.savefig(os.path.join(args.outdir, fname), dpi=150)
        plt.close(fig)
