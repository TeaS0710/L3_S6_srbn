import os
import sys
import matplotlib.pyplot as plt

# Imports depuis nos modules
from json_reader import collect_json_files, load_clusters_json
from plotter import plot_mds_2d

# On importe tqdm
from tqdm import tqdm

def main():
    # 1) Récupération du dossier JSON (soit en paramètre, soit par défaut)
    if len(sys.argv) > 1:
        JSON_DIR = sys.argv[1]
    else:
        JSON_DIR = "../../results"

    if not os.path.isdir(JSON_DIR):
        print(f"Répertoire introuvable: {JSON_DIR}")
        sys.exit(1)

    # 2) Récupérer tous les .json
    json_files = collect_json_files(JSON_DIR)
    if not json_files:
        print(f"Aucun fichier .json trouvé dans {JSON_DIR}. Fin.")
        sys.exit(0)

    # 3) Créer une figure multi-subplots
    n_files = len(json_files)
    n_cols = 2 if n_files > 1 else 1
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows), squeeze=False)
    ax_list = axes.ravel()

    # 4) Parcours des JSON avec tqdm
    print(f"Traitement de {n_files} fichiers JSON dans {JSON_DIR}...")
    for idx, json_path in enumerate(tqdm(json_files, desc="Progression")):
        ax = ax_list[idx]

        # Lecture
        tokens, sim_mat, clusters, file_origin = load_clusters_json(json_path)

        # Titre court
        base_name = os.path.basename(json_path)
        short_title = f"{file_origin}"

        # Appel du plot MDS 2D
        plot_mds_2d(ax, tokens, sim_mat,clusters, short_title)

    # Supprime les axes non utilisés si le nombre de fichiers est impair
    for j in range(idx+1, len(ax_list)):
        fig.delaxes(ax_list[j])

    plt.tight_layout()
    out_png = "../../results/compare_texts_2d.png"
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"[OK] Représentations 2D enregistrées dans: {out_png}")

if __name__ == "__main__":
    main()
