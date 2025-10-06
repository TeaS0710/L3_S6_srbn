#import des modules
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import DistanceMetric
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import json
import glob
import re
from collections import OrderedDict

#fonction qui lit un json et renvoie son contenu
def lire_fichier (chemin):
    with open(chemin) as json_data:  # ouvre fichier
        texte = json.load(json_data)  # charge en json
    return texte  # renvoie le contenu

#fonction pour récupérer proprement le nom du fichier
def nomfichier(chemin):
    nomfich = chemin.split("/")[-1].split(".")  # récupère nom sans le path
    return ("_").join([nomfich[0], nomfich[1]])  # reforme un nom propre

chemin_entree = ""  # chemin d'entrée non défini

# boucle sur les sous-corpus
for subcorpus in glob.glob(path_copora):
    liste_nom_fichier = []  # stocker les fichiers

    # cherche des fichiers json spécifiques
    for path in glob.glob("%s/AIMARD-TRAPPEURS_MOD/AIMARD_les-trappeurs_TesseractFra-PNG.txt_SEM_WiNER.ann_SEM.json-concat.json" % subcorpus):

        nom_fichier = nomfichier(path)  # extrait nom fichier
        liste = lire_fichier(path)  # lit fichier

        #### FREQUENCE ########
        dic_mots = {}  # init dico de fréquence
        i = 0  # compteur inutile mais bon

        # compte les occurrences des mots
        for mot in liste:
            dic_mots[mot] = dic_mots.get(mot, 0) + 1

        i += 1  # on incrémente au cas où

        # trie le dico par clé (alphabétique)
        new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

        # convertit liste en set puis en liste propre
        liste_words = [l for l in set(liste) if len(l) != 1]

        try:
            words = np.asarray(liste_words)  # met en numpy
            matrice = []  # init matrice de distance

            # boucle pour calculer les distances cosinus
            for w in words:
                liste_vecteur = []
                for w2 in words:
                    V = CountVectorizer(ngram_range=(2,3), analyzer='char')  # vectorisation
                    X = V.fit_transform([w, w2]).toarray()
                    distance = sklearn.metrics.pairwise.cosine_distances(X)[0][1]  # distance cosinus
                    liste_vecteur.append(distance)
                matrice.append(liste_vecteur)

            matrice_def = -1 * np.array(matrice)  # transformation pour clustering

            # clustering
            affprop = AffinityPropagation(affinity="precomputed", damping=0.6, random_state=None)
            affprop.fit(matrice_def)

            # récupère les clusters
            dic_output = {}
            for cluster_id in np.unique(affprop.labels_):
                exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
                cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])

                Id = "ID " + str(i)  # ID unique
                if exemplar in new_d:
                    dic_output[Id] = {
                        "Centroïde": exemplar,
                        "Freq. centroide": new_d[exemplar],
                        "Termes": cluster.tolist()
                    }

                i += 1  # incrémente

        except:  # si ça plante, on note le fichier
            print("**********Non OK***********", path)
            liste_nom_fichier.append(path)
            continue
