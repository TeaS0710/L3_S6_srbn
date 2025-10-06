import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

def exo2(csv_path='../../data/spam.csv', test_ratio=0.01):

    # Chargement des données
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # Suppression de colonnes
    df = df.iloc[:, :2]

    # Modification des headers et valeurs
    df.columns = ['labels', 'data']
    df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

    # Séparation entraînement / test
    df_train, df_test, y_train, y_test = train_test_split(
        df['data'], df['b_labels'], test_size=test_ratio
    )

    # Extraction de caractéristiques TF-IDF
    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)
    Xtest = tfidf.transform(df_test)

    # Naïve Bayes
    nb_model = MultinomialNB()
    nb_model.fit(Xtrain, y_train)
    score_nb = nb_model.score(Xtest, y_test)
    print(f"Précision modèle Bayes : {score_nb:.4f}")

    # AdaBoost
    ada_model = AdaBoostClassifier()
    ada_model.fit(Xtrain.toarray(), y_train)
    score_ada = ada_model.score(Xtest.toarray(), y_test)
    print(f"Précision modèle AdaBoost : {score_ada:.4f}")

    # On retourne le DataFrame pour le WordCloud
    return df

# Fonction WordCloud
def visualize(label, df):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=1920, height=1080).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f"WordCloud - {label}")
    plt.show()

if __name__ == '__main__':
    df = exo2()
    visualize('spam', df)
    visualize('ham', df)
