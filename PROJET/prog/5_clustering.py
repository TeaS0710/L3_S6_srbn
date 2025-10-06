import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    input_file = "../outputs/data_after_ner.json"
    output_file = "../outputs/clusters_2_3grams.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data_cleaned = json.load(f)

    all_words = []
    lang_labels = []

    # Récupération de tous les mots (lemmas filtrés) de chaque langue
    for lang_code, files_dict in data_cleaned.items():
        for file_name, lemmas in files_dict.items():
            for w in lemmas:
                all_words.append(w)
                lang_labels.append(lang_code)

    # Vectorisation n-grammes caractères 2 et 3
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
    X = vectorizer.fit_transform(all_words)

    # KMeans
    k = 3  # adapter si vous avez 3 langues, sinon ajuster
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    # Sauvegarde
    clusters_dict = {}
    for i, word in enumerate(all_words):
        clusters_dict[word] = {
            "cluster": int(cluster_labels[i]),
            "lang": lang_labels[i]
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clusters_dict, f, ensure_ascii=False, indent=2)

    # Visualisation PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.toarray())

    plt.figure()
    scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=cluster_labels)
    plt.title("Clustering (2-3 char n-grams)")
    plt.savefig("../outputs/clusters_2_3grams.png")
    plt.close()

if __name__ == "__main__":
    main()
