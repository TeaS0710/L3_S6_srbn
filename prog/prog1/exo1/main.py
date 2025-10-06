import glob
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN_PATH = "../../données/corpus_multi/*/appr/*"
TEST_PATH = "../../données/corpus_multi/*/test/*"
OUTPUT_PREDICTIONS = "../../resultats/exo1/predictions.json"
NGRAM_RANGE = (3, 3)
MAX_FEATURES = 5000
TOP_K = 3

def open_entrainementDataBase(base_path):
    files = glob.glob(base_path)
    data_by_lang = defaultdict(list)
    for file_path in files:
        parts = file_path.split(os.sep)
        lang = parts[-3]
        with open(file_path, "r", encoding="utf-8") as f:
            data_by_lang[lang].append(f.read())
    return data_by_lang

def ConstructionCalculs(data_by_lang, ngram_range=(3, 3), max_features=None):
    X_texts = []
    y_langs = []
    for lang, texts in data_by_lang.items():
        for t in texts:
            X_texts.append(t)
            y_langs.append(lang)
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(X_texts)
    X_array = X.toarray()
    centroids = {}
    for lang in data_by_lang:
        indices = [i for i, l in enumerate(y_langs) if l == lang]
        if len(indices) > 0:
            centroid = X_array[indices].mean(axis=0, keepdims=True)
        else:
            centroid = np.zeros((1, X_array.shape[1]))
        centroids[lang] = centroid
    return vectorizer, centroids

def PredictLangues(text, vectorizer, centroids, top_k=3):
    x = vectorizer.transform([text]).toarray()
    similarities = {}
    for lang, centroid in centroids.items():
        sim = cosine_similarity(x, centroid)[0, 0]
        similarities[lang] = sim
    sorted_langs = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    predicted_lang, best_sim = sorted_langs[0]
    total_sim = sum(similarities.values())
    confidence = best_sim / total_sim if total_sim > 0 else 0.0
    return predicted_lang, confidence, sorted_langs[:top_k]

def PyProgWork(centroids, vectorizer, test_path, top_k=3):
    test_files = glob.glob(test_path)
    predictions = {}
    for file_path in test_files:
        parts = file_path.split(os.sep)
        true_lang = parts[-3]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        pred_lang, confidence, top_langs = PredictLangues(text, vectorizer, centroids, top_k=top_k)
        predictions[file_path] = {
            "pred": pred_lang,
            "ref": true_lang,
            "confidence": confidence,
            "top_langs": top_langs
        }
    return predictions

def CalculPrediction(predictions):
    true_labels = []
    pred_labels = []
    for info in predictions.values():
        pred_labels.append(info["pred"])
        true_labels.append(info["ref"])
    all_langs = sorted(list(set(true_labels + pred_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_langs)
    report = classification_report(true_labels, pred_labels, labels=all_langs)
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels) if len(true_labels) else 0
    return cm, all_langs, report, accuracy

def PlotLangDist(centroids, all_langs):
    lang_vectors = []
    for lang in all_langs:
        lang_vectors.append(centroids[lang])
    lang_matrix = np.vstack(lang_vectors)
    sim_matrix = cosine_similarity(lang_matrix, lang_matrix)
    dist_matrix = 1 - sim_matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, xticklabels=all_langs, yticklabels=all_langs, cmap="magma", annot=False)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("../../resultats/exo1/languages_distance.png")
    plt.show()

def main():
    data_by_lang = open_entrainementDataBase(TRAIN_PATH)
    vectorizer, centroids = ConstructionCalculs(data_by_lang, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    predictions = PyProgWork(centroids, vectorizer, TEST_PATH, top_k=TOP_K)
    with open(OUTPUT_PREDICTIONS, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    cm, langs, report, accuracy = CalculPrediction(predictions)
    print("Matrice de confusion (lignes=réf, colonnes=prédiction) :")
    print(langs)
    print(cm)
    print("\nRapport de classification :")
    print(report)
    print(f"Exactitude globale : {accuracy:.2f}")
    PlotLangDist(centroids, langs)

if __name__ == "__main__":
    main()
