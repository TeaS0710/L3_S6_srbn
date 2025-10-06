import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as multi
from sklearn.ensemble import AdaBoostClassifier as ad

# Création de la fonction
def exo1(path='../../data/spambase.data'):

    # Chargement des données
    data = pd.read_csv(path, header=None).values
    np.random.shuffle(data)

    X = data[:, :-1]
    Y = data[:, -1]

    # Utilisation de la matrice
    Xtrain = X
    Ytrain = Y
    Xtest = X
    Ytest = Y

    # Modèle MultinomialNB
    nb_model = multi()
    nb_model.fit(Xtrain, Ytrain)
    nb_score = nb_model.score(Xtest, Ytest)
    print(f"Précision modèle MultinomialNB : {nb_score:.4f}")

    # Modèle AdaBoost
    ada_model = ad()
    ada_model.fit(Xtrain, Ytrain)
    ada_score = ada_model.score(Xtest, Ytest)
    print(f"Précision modèle AdaBoost : {ada_score:.4f}")

    # Affichage d'exemple
    print("mail :", X[0])
    print("(0 = ham, 1 = spam) :", Y[0])

    return {
        "MultinomialNB": nb_score,
        "AdaBoost": ada_score
    }

# Appel

if __name__ == '__main__':
    exo1()
