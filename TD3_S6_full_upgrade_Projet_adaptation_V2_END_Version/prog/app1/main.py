import os
import numpy as np
from tqdm import tqdm

# 📥 Lecture HTML multilingue
from reader import load_corpus_by_language

# 🔍 spaCy NER + lemmatisation
from spacy_processor import process_texts_by_lang

# 📊 Vectorisation et clustering
from vectorizer import vectorize_tokens, compute_similarity
from clusterer import run_affinity_propagation, build_clusters_dict

# 💾 Sauvegarde
from saver import save_result_for_file

# === Configuration ===
DATA_DIR = "../../corpus_multi"
RESULTS_DIR = "../../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NER_PREFIXES = ["B-LOC", "I-LOC"]
USE_POS = True
NGRAM_RANGE = (2, 3)
MIN_TOKENS = 5


def main():
    print("--- Étape 1 : Lecture du corpus HTML multilingue ---")
    corpus = load_corpus_by_language(DATA_DIR)
    print(f"[OK] Langues détectées : {list(corpus.keys())}")

    print("\n--- Étape 2 : Traitement spaCy (NER + Lemmatisation) ---")
    processed = process_texts_by_lang(
        corpus_dict=corpus,
        use_pos=USE_POS
    )

    print("\n--- Étape 3 : Clustering par langue ---")
    report = {}

    for lang, docs in tqdm(processed.items(), desc="Langues"):
        print(f"\n[INFO] Traitement de la langue : {lang}")
        all_tokens = []

        for doc in docs:
            tokens = [tok.lower() for tok, label in zip(doc["tokens"], doc["labels"]) if label in NER_PREFIXES]

            if not tokens:
                continue

            if USE_POS:
                all_tokens.extend([l for i, l in enumerate(doc["lemmes"]) if doc["labels"][i] in NER_PREFIXES])
            else:
                all_tokens.extend([
                    l.split("_")[0] for i, l in enumerate(doc["lemmes"])
                    if doc["labels"][i] in NER_PREFIXES and "_" in l
                ])

        print(f"[INFO] Total de tokens à clusteriser : {len(all_tokens)}")

        if len(all_tokens) < MIN_TOKENS:
            print(f"[WARN] Pas assez de lemmes pour clusteriser la langue '{lang}'.")
            report[lang] = {"status": "❌ insuffisant", "n_tokens": len(all_tokens)}
            continue

        X, vect = vectorize_tokens(all_tokens, analyzer="char", ngram_range=NGRAM_RANGE)
        similarity_matrix = compute_similarity(X, metric="cosine")
        labels, centers_idx = run_affinity_propagation(similarity_matrix)
        clusters_dict = build_clusters_dict(labels, centers_idx, all_tokens)

        out_json_name = f"clusters_{lang}_ngrams_{NGRAM_RANGE[0]}_{NGRAM_RANGE[1]}.json"
        out_json_path = os.path.join(RESULTS_DIR, out_json_name)

        save_result_for_file(
            filepath=f"<lang={lang}>",
            similarity_matrix=similarity_matrix,
            clusters_dict=clusters_dict,
            out_json_path=out_json_path,
            used_tokens=all_tokens
        )

        report[lang] = {"status": "✅ OK", "n_tokens": len(all_tokens)}

    print("\n--- ✅ Résumé du traitement ---")
    total = len(report)
    done = sum(1 for r in report.values() if r["status"] == "✅ OK")
    skipped = total - done

    for lang, info in report.items():
        print(f"  {lang:<10} → {info['status']} ({info['n_tokens']} tokens)")

    print(f"\nLangues traitées : {done}/{total}  |  Ignorées : {skipped}")


if __name__ == "__main__":
    main()
