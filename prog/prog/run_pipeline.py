import os
from html_loader import load_corpus_by_language, save_corpus_to_json
from spacy_processor_multilang import process_texts_by_lang, save_processed_data
from cluster_multilang import cluster_all_languages
from visualize_stats import main as plot_stats_main
from visualize_clusters import visualize_all_clusters

def run_full_pipeline(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Étape 1 : extraction HTML
    corpus_json = os.path.join(output_dir, "corpus_grouped_by_lang.json")
    print("\n--- Étape 1 : Extraction HTML ---")
    corpus = load_corpus_by_language(base_dir)
    save_corpus_to_json(corpus, corpus_json)

    # Étape 2 : traitement linguistique
    processed_json = os.path.join(output_dir, "processed_multilang.json")
    print("\n--- Étape 2 : Lemmatisation + NER + Stats ---")
    processed = process_texts_by_lang(corpus)
    save_processed_data(processed, processed_json)

    # Étape 3 : Clustering (n-grammes)
    print("\n--- Étape 3 : Clustering bigrammes/trigrammes ---")
    cluster_all_languages(
        input_path=processed_json,
        output_path=os.path.join(output_dir, "clusters_ngrams_2_3.json"),
        ngram_range=(2, 3)
    )

    print("\n--- Étape 4 : Clustering 4-5-grammes ---")
    cluster_all_languages(
        input_path=processed_json,
        output_path=os.path.join(output_dir, "clusters_ngrams_4_5.json"),
        ngram_range=(4, 5)
    )

    # Étape 5 : Visualisation statistiques linguistiques
    print("\n--- Étape 5 : Visualisation statistiques ---")
    plot_stats_main()

    # Étape 6 : Visualisation des clusters
    print("\n--- Étape 6 : Visualisation des clusters ---")
    visualize_all_clusters(
        json_path=os.path.join(output_dir, "clusters_ngrams_2_3.json"),
        title_prefix="Clusters lexicaux (bi/tri-grammes)"
    )
    visualize_all_clusters(
        json_path=os.path.join(output_dir, "clusters_ngrams_4_5.json"),
        title_prefix="Clusters lexicaux (4/5-grammes)"
    )

    print("\n✅ Pipeline terminé avec succès !")

# Lancer le pipeline
if __name__ == "__main__":
    BASE_DIR = "../corpus_multi"
    OUTPUT_DIR = "../pipeline_results"

    run_full_pipeline(BASE_DIR, OUTPUT_DIR)
