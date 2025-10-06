import os
import glob

def parse_iob_file(filepath):
    """
    Lit un fichier .bio/.iob2 supposant au moins 2 colonnes : (token, label).
    Retourne une liste de tuples (token, label).
    """
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Ex : parts[0] = token, parts[-1] = label NER (B-LOC, I-LOC, ...)
            token = parts[0]
            label = parts[-1]
            items.append((token, label))
    return items
