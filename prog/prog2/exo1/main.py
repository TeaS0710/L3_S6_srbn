import glob
import spacy
import json
import matplotlib.pyplot as plt
import numpy as np
import os

### Modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

AUTEURS = ["DAUDET", "MAUPASSANT"]
OUTPUT_DIR = "../../resultats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def freq_dict(liste):
    """Construit un dict de fréquences pour chaque élément."""
    d = {}
    for x in liste:
        d[x] = d.get(x, 0) + 1
    return d

def top_n(dico, n=10):
    """Retourne les n éléments les plus fréquents d'un dict."""
    items_tries = sorted(dico.items(), key=lambda x: x[1], reverse=True)
    return items_tries[:n]

def bar_chart_top_n(elements_freq, titre, chemin_sortie):
    """Trace un diagramme en barres pour les paires (élément, fréquence)."""
    if not elements_freq:
        return
    labels = [x[0] for x in elements_freq]
    valeurs = [x[1] for x in elements_freq]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, valeurs, color='cornflowerblue')
    plt.title(titre)
    plt.xlabel("Éléments")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.show()
    plt.close()

def histogram_token_lengths(token_lengths, titre, chemin_sortie):
    """Trace un histogramme de la distribution de la longueur des tokens."""
    if not token_lengths:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(token_lengths, bins=range(1, max(token_lengths)+2), color='skyblue', edgecolor='black')
    plt.title(titre)
    plt.xlabel("Longueur du token")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(chemin_sortie)
    plt.show()
    plt.close()

def analyze_text(texte):
    """Analyse le texte avec spaCy et calcule des mesures."""
    doc = nlp(texte)
    nb_tokens = len(doc)
    nb_caracteres = len(texte)
    lemmes = []
    pos_tags = []
    token_lengths = []
    for token in doc:
        if not token.is_punct:
            lemmes.append(token.lemma_)
            pos_tags.append(token.pos_)
            token_lengths.append(len(token.text))
    return {
        "freq_lemmes": freq_dict(lemmes),
        "freq_pos": freq_dict(pos_tags),
        "token_lengths": token_lengths,
        "nb_tokens": nb_tokens,
        "nb_caracteres": nb_caracteres,
        "moyenne_longueur_token": np.mean(token_lengths) if token_lengths else 0
    }

def process_file(chemin_fichier, auteur_dir):
    """Analyse un fichier texte, génère et enregistre les graphiques individuels."""
    nom_fichier_court = os.path.basename(chemin_fichier)
    with open(chemin_fichier, "r", encoding="utf-8") as fin:
        texte = fin.read()
    stats = analyze_text(texte)
    top_10_lemmes = top_n(stats["freq_lemmes"], 10)
    top_10_pos = top_n(stats["freq_pos"], 10)
    bar_chart_top_n(top_10_lemmes, f"Top 10 lemmes ({nom_fichier_court})", os.path.join(auteur_dir, f"lemmes_{nom_fichier_court}.png"))
    bar_chart_top_n(top_10_pos, f"Top 10 POS ({nom_fichier_court})", os.path.join(auteur_dir, f"pos_{nom_fichier_court}.png"))
    histogram_token_lengths(stats["token_lengths"], f"Distribution des longueurs de tokens ({nom_fichier_court})", os.path.join(auteur_dir, f"hist_token_length_{nom_fichier_court}.png"))
    return {
        "nb_tokens": stats["nb_tokens"],
        "nb_caracteres": stats["nb_caracteres"],
        "moyenne_longueur_token": stats["moyenne_longueur_token"],
        "lemmes": stats["freq_lemmes"],
        "pos": stats["freq_pos"]
    }

def process_author(auteur):
    """Traite tous les fichiers .txt d'un auteur et agrège les statistiques."""
    fichiers = glob.glob(f"../../données/ressources_TD1_Entite-nommee/Texte/{auteur}/**/*.txt", recursive=True)
    auteur_dir = os.path.join(OUTPUT_DIR, auteur)
    os.makedirs(auteur_dir, exist_ok=True)
    resultats_auteur = {}
    fichiers_list, nb_tokens_list, nb_carac_list = [], [], []
    for chemin_fichier in fichiers:
        nom_fichier_court = os.path.basename(chemin_fichier)
        file_stats = process_file(chemin_fichier, auteur_dir)
        resultats_auteur[nom_fichier_court] = file_stats
        fichiers_list.append(nom_fichier_court)
        nb_tokens_list.append(file_stats["nb_tokens"])
        nb_carac_list.append(file_stats["nb_caracteres"])
    with open(os.path.join(auteur_dir, f"resultats_{auteur}.json"), "w", encoding="utf-8") as fout:
        json.dump(resultats_auteur, fout, ensure_ascii=False, indent=4)
    return {
        "fichiers": fichiers_list,
        "nb_tokens": nb_tokens_list,
        "nb_caracteres": nb_carac_list
    }

def generate_global_charts(auteur, stats):
    """Génère des graphiques comparatifs pour un auteur."""
    fichiers_auteur = stats["fichiers"]
    nb_tokens_auteur = stats["nb_tokens"]
    nb_caracteres_auteur = stats["nb_caracteres"]
    auteur_dir = os.path.join(OUTPUT_DIR, auteur)
    if not fichiers_auteur:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(fichiers_auteur, nb_tokens_auteur, color='lightgreen')
    plt.title(f"Nombre de tokens par fichier - {auteur}")
    plt.xlabel("Fichier")
    plt.ylabel("Nombre de tokens")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(auteur_dir, f"nb_tokens_{auteur}.png"))
    plt.show()
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.bar(fichiers_auteur, nb_caracteres_auteur, color='salmon')
    plt.title(f"Nombre de caractères par fichier - {auteur}")
    plt.xlabel("Fichier")
    plt.ylabel("Nombre de caractères")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(auteur_dir, f"nb_caracteres_{auteur}.png"))
    plt.show()
    plt.close()

def main():
    """Lance l'analyse globale pour chaque auteur."""
    stats_par_auteur = {}
    for auteur in AUTEURS:
        stats_par_auteur[auteur] = process_author(auteur)
    for auteur in AUTEURS:
        generate_global_charts(auteur, stats_par_auteur[auteur])
    print("[INFO] Analyse terminée.")

if __name__ == "__main__":
    main()
