# main.py (orchestration)
import subprocess
from os import sys
import os

def main():
    os.makedirs("../outputs", exist_ok=True)

    # 1. Load HTML -> data_loaded.json
    subprocess.run([sys.executable, "script_load_html.py"], check=True)

    # 2. Preprocessing by language
    data = load_data("../outputs/data_loaded.json")
    langs = data.keys()
    preprocessed_files = []
    for lang in langs:
        out_path = f"../outputs/preprocessed_{lang}.json"
        preprocess_language(data, lang, out_path)
        preprocessed_files.append(out_path)

    # 3. Stats
    compute_stats(preprocessed_files, "../outputs/stats_all.csv")

    # 4. Visualisation
    plot_stats("../outputs/stats_all.csv", "../outputs/plots_all")

    # 5. Clustering
    all_tokens = load_all_tokens(preprocessed_files)
    cluster_tokens(all_tokens, (2,3), "../outputs/clusters_2-3gram")
    cluster_tokens(all_tokens, (4,5), "../outputs/clusters_4-5gram")

if __name__ == "__main__":
    main()
