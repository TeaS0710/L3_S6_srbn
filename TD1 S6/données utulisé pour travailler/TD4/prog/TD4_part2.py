import glob
import json
import os
from collections import defaultdict, Counter

input = r"..\corpus_multi\*\test\*"

def open_file(file_path="models.json"): # Charger les modèles sauvegardés
    with open(file_path, "r", encoding="utf-8") as fichier:
        models = json.load(fichier)
    return models

def prediction_de_langue(dico_test, models): # Fonction pour prédire la langue d'un fichier de test
    max_intersection = 0
    prediction = None

    for langue, mots_frequents in models.items():
        intersection = len(set(dico_test.keys()) & set(mots_frequents))
        if intersection > max_intersection:
            max_intersection = intersection
            prediction = langue
    return prediction

def process_test_files(models): # Fonction pour traiter les fichiers de test et récupérer les prédictions
    predictions = {}
    liste_fichier_test = glob.glob(input)
    print(f"Nombre de fichier de test : {len(liste_fichier_test)}")

    for chemin in liste_fichier_test:
        print(f"Traitement du fichier de test : {chemin}")

        dossiers = chemin.split("\\") # Obtenir la langue réelle à partir du chemin du fichier
        langue_reelle = dossiers[2]

        with open(chemin, "r", encoding="utf-8") as fichier: # Extraire les mots fréquents du fichier de test
            mots = fichier.read().split()
        dico_test = dict(Counter(mots).most_common(10))
        langue_predite = prediction_de_langue(dico_test, models) # Prédire la langue
        predictions[chemin] = {"prediction": langue_predite, "reference": langue_reelle}

    return predictions



def evaluer_statistiques(predictions): # Calcul des statistiques d'évaluation
    nb_bonnes_reponses = sum(1 for result in predictions.values() if result["prediction"] == result["reference"])
    total = len(predictions)
    exactitude = nb_bonnes_reponses / total if total > 0 else 0

    stats_par_langue = defaultdict(lambda: {"VP": 0, "FP": 0, "FN": 0})

    for result in predictions.values():
        langue_reelle = result["reference"]
        langue_predite = result["prediction"]
        if langue_predite == langue_reelle:
            stats_par_langue[langue_reelle]["VP"] += 1
        else:
            stats_par_langue[langue_reelle]["FN"] += 1
            stats_par_langue[langue_predite]["FP"] += 1

    resultats_final = {}
    for langue, stats in stats_par_langue.items():
        vp = stats["VP"]
        fp = stats["FP"]
        fn = stats["FN"]

        rappel = vp / (vp + fn) if (vp + fn) > 0 else 0 #silence
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0 #bruit
        f1_mesure = (2 * rappel * precision) / (precision + rappel) if (precision + rappel) > 0 else 0 #perfromance

        resultats_final[langue] = {
            "rappel": rappel,
            "precision": precision,
            "f1_mesure": f1_mesure
        }

    print(f"Exactitude globale : {exactitude}")
    for langue, stats in resultats_final.items():
        print(f"Langue : {langue}")
        print(f" - Rappel : {stats['rappel']}")
        print(f" - Précision : {stats['precision']}")
        print(f" - F1-mesure : {stats['f1_mesure']}")

    return exactitude, resultats_final


# Fonction principale pour l'exercice 3
def main():
    models = open_file()
    predictions = process_test_files(models)
    exactitude, stats_par_langue = evaluer_statistiques(predictions)


    with open("predictions.json", "w", encoding="utf-8") as fichier: # Sauvegarder les résultats dans un fichier JSON
        json.dump(predictions, fichier, indent=2, ensure_ascii=False)
    print("Prédictions sauvegardées dans predictions.json")


main()
