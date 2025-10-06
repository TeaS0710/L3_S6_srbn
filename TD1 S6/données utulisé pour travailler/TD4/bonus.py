from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import glob
import os

# Initialiser les dictionnaires pour stocker les textes et les classes
texte = {"appr": [], "test": []}
classes = {"appr": [], "test": []}

# Parcourir les fichiers pour extraire les textes et les classes
for path in glob.glob(r"C:\Users\vergn\Desktop\PROG S5\TD4\corpus_multi\cs\appr\2009-01-15_celex_IP-09-57.cs.html")[:1500]:
    # Obtenir la langue et le corpus (appr ou test)
    path_parts = path.split(os.sep)
    lang, corpus = path_parts[-3], path_parts[-2]

    if corpus in texte:  # S'assurer qu'on est bien dans "appr" ou "test"
        classes[corpus].append(lang)
        with open(path, 'r', encoding="utf-8") as f:
            texte[corpus].append(f.read())

# Initialiser le CountVectorizer après avoir collecté tous les textes
vectorizer = CountVectorizer(max_features=1500)

# Créer les matrices de caractéristiques pour les ensembles d'apprentissage et de test
Xtrain = vectorizer.fit_transform(texte["appr"]).toarray()
Xtest = vectorizer.transform(texte["test"]).toarray()
ytrain = classes["appr"]
ytest = classes["test"]

# Initialiser et entraîner le classifieur Naive Bayes, puis faire des prédictions
gnb = GaussianNB()
ypred = gnb.fit(Xtrain, ytrain).predict(Xtest)

# Afficher les résultats
print("Erreurs d'étiquetage sur %d textes: %d" % (Xtest.shape[0], (ytest != ypred).sum()))
print(classification_report(ytest, ypred))
