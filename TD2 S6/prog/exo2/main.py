import glob
import spacy
import re
import json
import os
import matplotlib.pyplot as plt

nlp = spacy.load("fr_core_news_sm")
AUTEURS = ["DAUDET", "MAUPASSANT"]
RESULTS_DIR = "../../resultats"
os.makedirs(RESULTS_DIR, exist_ok=True)

def freq_dict(tokens):
    d = {}
    for t in tokens:
        d[t] = d.get(t, 0) + 1
    return d

def zipf_plot(freq_s, freq_sp, auteur):
    if not freq_s or not freq_sp:
        return
    vs = sorted(freq_s.values(), reverse=True)
    vp = sorted(freq_sp.values(), reverse=True)
    plt.loglog(range(1, len(vs) + 1), vs, label='split')
    plt.loglog(range(1, len(vp) + 1), vp, label='spaCy')
    plt.title(f'Loi de Zipf - {auteur}')
    plt.xlabel('Rang')
    plt.ylabel('Fréquence')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'zipf_{auteur}.png'))
    plt.show()
    plt.close()

def analyze_files(fichiers):
    res = {}
    for f in fichiers:
        with open(f, encoding="utf-8") as fin:
            texte = fin.read()
        tokens_split = texte.split()
        tokens_spacy = [tok.text for tok in nlp(texte)]
        res[f] = {
            "split": freq_dict(tokens_split),
            "spacy": freq_dict(tokens_spacy)
        }
    return res

def process_author(auteur):
    fichiers = glob.glob(f"../../données/ressources_TD1_Entite-nommee/Texte/{auteur}/**/*.txt", recursive=True)
    ref_files = [f for f in fichiers if re.search(r"kraken", f, re.IGNORECASE)]
    cible_files = [f for f in fichiers if re.search(r"_ref", f, re.IGNORECASE)]
    res_ref = analyze_files(ref_files)
    res_cible = analyze_files(cible_files)
    with open(os.path.join(RESULTS_DIR, f"resultats_{auteur}_ref.json"), "w", encoding="utf-8") as fout:
        json.dump(res_ref, fout, ensure_ascii=False, indent=4)
    with open(os.path.join(RESULTS_DIR, f"resultats_{auteur}_cible.json"), "w", encoding="utf-8") as fout:
        json.dump(res_cible, fout, ensure_ascii=False, indent=4)
    for f in res_ref:
        zipf_plot(res_ref[f]["split"], res_ref[f]["spacy"], auteur)

def main():
    for auteur in AUTEURS:
        process_author(auteur)
    print("Analyse terminée")

if __name__ == "__main__":
    main()
