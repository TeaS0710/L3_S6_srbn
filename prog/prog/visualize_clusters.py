import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

def load_clusters(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_clusters_for_lang(lang, clusters_dict, title):
    labels = []
    tokens = []

    for cid, data in clusters_dict.items():
        for token in data["members"]:
            tokens.append(token)
            labels.append(int(cid))

    if len(set(labels)) < 2:
        print(f"[WARN] Trop peu de clusters pour {lang}, skipping.")
        return

    # Vectorisation TF-IDF + Réduction de dimension
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    X = vectorizer.fit_transform(tokens)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Affichage
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title(f"{title} – {lang}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_all_clusters(json_path, title_prefix):
    all_clusters = load_clusters(json_path)

    for lang, clusters_dict in all_clusters.items():
        plot_clusters_for_lang(lang, clusters_dict, title=f"{title_prefix} ({lang})")

# Exemple d'exécution
if __name__ == "__main__":
    visualize_all_clusters(
        json_path="../pipeline_results/clusters_ngrams_2_3.json",
        title_prefix="Clusters lexicaux (bi/tri-grammes)"
    )

    visualize_all_clusters(
        json_path="../pipeline_results/clusters_ngrams_4_5.json",
        title_prefix="Clusters lexicaux (4/5-grammes)"
    )
