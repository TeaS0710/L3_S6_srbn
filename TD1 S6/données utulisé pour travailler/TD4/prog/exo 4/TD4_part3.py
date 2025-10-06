import glob
import json

input = r"..\..\corpus_multi\*\appr\*"
# Fonction pour obtenir la liste des fichiers
def openfile(input):
    liste_fichier = glob.glob(input)
    print(f"Nombre de fichiers : {len(liste_fichier)}")
    return liste_fichier


# Fonction pour construire le dictionnaire de mots par langue
def dico(liste_fichier):
    dic_langues = {}
    for chemin in liste_fichier:
        print(f"Traitement du fichier : {chemin}")

        dossiers = chemin.split("\\") # Diviser le chemin et extraire le nom du dossier de langue
        print(dossiers)
        langue = dossiers[3]  # Assurez-vous que c'est bien l'index correct pour le nom de langue

        if langue not in dic_langues:
            dic_langues[langue] = {}

        with open(chemin, "r", encoding="utf-8") as fichier: # Lire le contenu du fichier
            chaine = fichier.read()

        mots = chaine.split() #tokeniser les mots ayant un espace avant et aprés

        for mot in mots:
            if mot not in dic_langues[langue]:
                dic_langues[langue][mot] = 1
            else:
                dic_langues[langue][mot] += 1

    print("Dictionnaire des langues :", dic_langues)
    return dic_langues


# Fonction pour trouver les 10 mots les plus fréquents par langue
def maxmots(dic_langues):
    dic_model = {}
    for langue, dic_effectifs in dic_langues.items():
        paires = [[mot, effectif] for mot, effectif in dic_effectifs.items()]
        liste_tri = sorted(paires, key=lambda x: x[1], reverse=True)[:10]  # Garder les 10 plus fréquents
        dic_model[langue] = [mot for mot, effetif in liste_tri]
        print(f"Top mots pour la langue {langue} : {dic_model[langue]}")
    return dic_model


# Fonction pour sauvegarder le dictionnaire final dans un fichier JSON
def jsonation(dic_model):
    with open("models.json", "w", encoding="utf-8") as fichier:
        json.dump(dic_model, fichier, indent=2, ensure_ascii=False)
    print("Dictionnaire sauvegardé dans models.json")


# Fonction principale
def main():
    liste_fichier = openfile(input)
    dic_langues = dico(liste_fichier)
    dic_model = maxmots(dic_langues)
    jsonation(dic_model)


main()
