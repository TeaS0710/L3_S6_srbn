#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

def vectorize_tokens(tokens, analyzer='char', ngram_range=(2,3)):
    """
    Vectorise la liste de tokens en utilisant CountVectorizer
    (par défaut, n-grammes de caractères).

    Retourne la matrice X (sparse) et l'objet vectorizer.
    """
    vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X = vectorizer.fit_transform(tokens)
    return X, vectorizer

def compute_similarity(X, metric='cosine'):
    """
    Calcule la matrice de similarité à partir de la matrice X.
    (distance cosinus => similarité = 1 - distance)
    """
    dist_matrix = pairwise_distances(X, metric=metric)
    similarity = 1.0 - dist_matrix
    return similarity
