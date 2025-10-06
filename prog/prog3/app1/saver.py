import json

def save_result_for_file(
    filepath,
    similarity_matrix,
    clusters_dict,
    out_json_path,
    used_tokens
):
    """
    Sauvegarde dans un seul JSON :
      - le chemin du fichier source
      - la liste des tokens utilisés (used_tokens)
      - la matrice de similarité (liste de listes)
      - l'objet 'clusters' (dictionnaire)
    """
    mat_list = similarity_matrix.tolist()
    data_out = {
        "file": filepath,
        "tokens": used_tokens,    # nouvelle clé
        "similarity_matrix": mat_list,
        "clusters": clusters_dict
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=4)
