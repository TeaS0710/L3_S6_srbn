# MODEL
from typing import List, Tuple
import numpy as np


def compute_counts(texts_int: List[List[int]], V: int,) -> Tuple[np.ndarray, np.ndarray]:

    A = np.ones((V, V), dtype=np.float64)
    pi = np.ones(V, dtype=np.float64)

    for sent in texts_int:
        if not sent:
            continue
        pi[sent[0]] += 1
        for u, v in zip(sent[:-1], sent[1:]):
            A[u, v] += 1
    return A, pi


def normalise(matrix_or_vec: np.ndarray) -> np.ndarray:

    if matrix_or_vec.ndim == 1:
        return matrix_or_vec / matrix_or_vec.sum()
    # matrice
    row_sums = matrix_or_vec.sum(axis=1, keepdims=True)
    return matrix_or_vec / row_sums


def build_markov_models(train_int: List[List[int]], y_train: List[int], V: int):

    classes = sorted(set(y_train))
    logAs, logPis, logPriors = [], [], []

    total = len(y_train)

    for c in classes:
        class_texts = [t for t, y in zip(train_int, y_train) if y == c]
        A, pi = compute_counts(class_texts, V)
        A = normalise(A)
        pi = normalise(pi)
        logAs.append(np.log(A))
        logPis.append(np.log(pi))
        prior = sum(1 for y in y_train if y == c) / total
        logPriors.append(np.log(prior))

    return logAs, logPis, logPriors
