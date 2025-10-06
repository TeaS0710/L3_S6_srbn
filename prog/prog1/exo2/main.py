"""
Script : NER au format IOB (Version avec pandas + tqdm pour observer la progression)

1. Extraire tous les fichiers .csv et .txt d'un dossier (et sous-dossiers).
2. Convertir les CSV annotés en IOB (gold standard).
3. Annoter les fichiers texte avec spaCy au format IOB.
4. Aligner et comparer les annotations pour calculer F1.
5. Sauvegarder les résultats et afficher le temps de traitement.
"""

import os
import time
import spacy
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm  # On importe tqdm pour les barres de progression

CHEMIN_DOSSIER = "../../données/ressources_TD1_Entite-nommee"  # Dossier contenant vos .csv et .txt
MODELE_SPACY = "en_core_web_sm"  # Adapter si nécessaire (ex: "fr_core_news_sm")
CHEMIN_SORTIE_IOB = "../../resultats/exo2/IOB_outputs"  # Nouveau dossier de sortie pour les fichiers .bio

# Si vous voulez séparer « gold » et « auto » en deux sous-dossiers
# vous pouvez décommenter et les utiliser.
# CHEMIN_SORTIE_GOLD = os.path.join(CHEMIN_SORTIE_IOB, "gold")
# CHEMIN_SORTIE_AUTO = os.path.join(CHEMIN_SORTIE_IOB, "auto")


def ListAllFiles(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    Parcourt récursivement le dossier `root_dir` pour récupérer :
      - la liste de tous les chemins des .csv
      - la liste de tous les chemins des .txt
    Les autres fichiers sont ignorés.
    """
    csv_files = []
    txt_files = []
    for chemin_racine, dirs, fichiers in os.walk(root_dir):
        for f in fichiers:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(chemin_racine, f))
            elif f.lower().endswith(".txt"):
                txt_files.append(os.path.join(chemin_racine, f))
    return csv_files, txt_files


def CsvToIob(csv_file: str) -> List[Tuple[str, str, str]]:
    """
    Lit un fichier CSV (avec colonnes : Token, LOC, PER, ORG, MISC) grâce à pandas.
    - 'Token' contient le token
    - Chacune des colonnes (LOC, PER, ORG, MISC) peut contenir :
        '' pour dire "pas d'entité" (ou 'O'),
        'B' (begin) ou 'I' (inside),
      ou éventuellement 'B-LOC', 'I-LOC', etc. si c'est déjà codé complet (à adapter).

    Retourne une liste de tuples : (token, iob_tag, ent_type).
    Exemple : [("San", "B", "LOC"), ("Francisco", "I", "LOC"), ...]
    """
    iob_data = []
    try:
        # Lecture "tolérante" : on laisse pandas deviner le séparateur
        df = pd.read_csv(
            csv_file,
            sep=None,  # on tente de deviner le séparateur
            engine='python',
            dtype=str,
            keep_default_na=False,  # évite de transformer '' en NaN
            na_values=[]
        )

        required_cols = ["Token", "LOC", "PER", "ORG", "MISC"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne '{col}' introuvable dans {csv_file}")

        for _, row in df.iterrows():
            # Extraction des champs
            token = (row["Token"] or "").strip()
            val_loc = (row["LOC"] or "").strip()
            val_per = (row["PER"] or "").strip()
            val_org = (row["ORG"] or "").strip()
            val_misc = (row["MISC"] or "").strip()

            # Par défaut, rien n'est annoté
            iob_tag = "O"
            ent_type = "O"

            # On vérifie une par une : LOC > PER > ORG > MISC
            if val_loc:
                iob_tag = val_loc
                ent_type = "LOC"
            elif val_per:
                iob_tag = val_per
                ent_type = "PER"
            elif val_org:
                iob_tag = val_org
                ent_type = "ORG"
            elif val_misc:
                iob_tag = val_misc
                ent_type = "MISC"

            iob_data.append((token, iob_tag, ent_type))

    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier {csv_file} : {e}")

    return iob_data


def SpacyWork(text_file: str, nlp) -> List[Tuple[str, str, str]]:
    """
    Lit un fichier texte brut, l'annote avec spaCy, et le retourne au format IOB.
    Renvoie la liste de tuples (token, IOB, ent_type).
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = nlp(text)
    iob_data = []
    for token in doc:
        iob_tag = token.ent_iob_
        ent_type = token.ent_type_ if token.ent_type_ else "O"
        iob_data.append((token.text, iob_tag, ent_type))
    return iob_data


def SaveIob(iob_data: List[Tuple[str, str, str]], output_path: str) -> None:
    """
    Sauvegarde la liste de tuples (token, iob, ent_type) au format IOB
    dans `output_path`, avec tabulation comme séparateur.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, iob, ent_type in iob_data:
            f.write(f"{token}\t{iob}\t{ent_type}\n")


def align_and_evaluate(
    gold_iob: List[Tuple[str, str, str]],
    auto_iob: List[Tuple[str, str, str]]
) -> Tuple[float, float, float]:
    """
    Alignement naïf des listes gold_iob et auto_iob pour calculer
    précision, rappel et F1. Suppose un alignement 1:1 par index.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    len_min = min(len(gold_iob), len(auto_iob))
    for i in range(len_min):
        gold_token, gold_iob_tag, gold_type = gold_iob[i]
        auto_token, auto_iob_tag, auto_type = auto_iob[i]

        # On compare (iob_tag, ent_type)
        if (gold_iob_tag != 'O') or (auto_iob_tag != 'O'):
            if (
                (gold_iob_tag == auto_iob_tag)
                and (gold_type == auto_type)
                and (gold_iob_tag != 'O')
            ):
                true_positives += 1
            else:
                if auto_iob_tag != 'O':
                    false_positives += 1
                if gold_iob_tag != 'O':
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return precision, recall, f1


def main():
    start_time = time.time()

    # 1) Chargement du modèle spaCy
    nlp = spacy.load(MODELE_SPACY)

    # 2) Récupération des chemins de fichiers
    csv_files, txt_files = ListAllFiles(CHEMIN_DOSSIER)
    os.makedirs(CHEMIN_SORTIE_IOB, exist_ok=True)

    # Pour séparer gold/auto, décommentez :
    # os.makedirs(CHEMIN_SORTIE_GOLD, exist_ok=True)
    # os.makedirs(CHEMIN_SORTIE_AUTO, exist_ok=True)

    # 3) Conversion des CSV annotés (gold) au format IOB et sauvegarde
    gold_annotations = {}
    print("Conversion des CSV annotés en IOB (gold) :")
    for csv_path in tqdm(csv_files, desc="CSV -> IOB"):
        csv_iob = CsvToIob(csv_path)

        # Nom de fichier de sortie (ex: "file.gold.bio")
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        out_file_name = f"{base_name}.gold.bio"

        # Si vous souhaitez tout mettre dans un seul dossier :
        out_path = os.path.join(CHEMIN_SORTIE_IOB, out_file_name)

        # Si vous vouliez un sous-dossier "gold" :
        # out_path = os.path.join(CHEMIN_SORTIE_GOLD, out_file_name)

        SaveIob(csv_iob, out_path)
        gold_annotations[csv_path] = csv_iob

    # 4) Annotation automatique des fichiers texte
    print("\nAnnotation automatique des fichiers texte avec spaCy :")
    auto_annotations = {}
    for txt_path in tqdm(txt_files, desc="TXT -> IOB (spaCy)"):
        spacy_iob = SpacyWork(txt_path, nlp)

        # Nom de fichier de sortie (ex: "file.auto.bio")
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        out_file_name = f"{base_name}.auto.bio"

        # Si vous souhaitez tout mettre dans le même dossier :
        out_path = os.path.join(CHEMIN_SORTIE_IOB, out_file_name)

        # Si vous vouliez un sous-dossier "auto" :
        # out_path = os.path.join(CHEMIN_SORTIE_AUTO, out_file_name)

        SaveIob(spacy_iob, out_path)
        auto_annotations[txt_path] = spacy_iob

    # 5) Comparaison CSV vs TXT (naïf : base de nom identique)
    print("\nComparaison (Gold vs Auto) et calcul des métriques :")
    evaluation_results = []
    for csv_file in tqdm(gold_annotations, desc="Évaluation"):
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        for txt_file in auto_annotations:
            txt_basename = os.path.splitext(os.path.basename(txt_file))[0]
            if txt_basename == csv_basename:
                gold_iob = gold_annotations[csv_file]
                auto_iob = auto_annotations[txt_file]
                precision, recall, f1 = align_and_evaluate(gold_iob, auto_iob)
                evaluation_results.append((csv_basename, precision, recall, f1))
                # On sort du for, car on ne compare qu'une fois
                break

    # 6) Affichage des scores
    print("\n=== Évaluation (CSV gold vs TXT auto) ===")
    for basename, precision, recall, f1 in evaluation_results:
        print(f"Fichier : {basename}")
        print(f"  Précision : {precision:.3f}")
        print(f"  Rappel    : {recall:.3f}")
        print(f"  F1-score  : {f1:.3f}\n")

    # 7) Temps de traitement total
    total_time = time.time() - start_time
    print(f"Temps total de traitement : {total_time:.2f} secondes")


if __name__ == "__main__":
    main()
