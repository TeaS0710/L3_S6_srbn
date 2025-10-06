#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np

def collect_json_files(json_folder):
    """
    Récupère la liste des fichiers JSON présents dans 'json_folder'.
    Retourne une liste de chemins absolus/relatifs.
    """
    pattern = os.path.join(json_folder, "*.json")
    filepaths = glob.glob(pattern)
    return filepaths

def load_clusters_json(json_path):
    """
    Lit un JSON de clusters contenant par ex.:
    {
      "file": "...",
      "tokens": [...],
      "similarity_matrix": [...],
      "clusters": { "0": {...}, "1": {...}, ... }
    }
    Retourne (tokens, similarity_matrix, clusters, file_origin).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = data["tokens"]  # liste de chaînes
    sim_mat = np.array(data["similarity_matrix"])  # np.array NxN
    clusters = data["clusters"]                    # dict
    file_origin = data.get("file", os.path.basename(json_path))

    return tokens, sim_mat, clusters, file_origin
