#CLASS
from typing import List
import numpy as np


class BigramNBClassifier:
    def __init__(self, logAs, logPis, logPriors):
        self.logAs = logAs
        self.logPis = logPis
        self.logPriors = logPriors
        self.n_classes = len(logAs)

    def _log_likelihood(self, tokens: List[int], cls: int) -> float:
        logA = self.logAs[cls]
        logPi = self.logPis[cls]

        if not tokens:
            return -np.inf  # phrase vide

        ll = logPi[tokens[0]]
        for u, v in zip(tokens[:-1], tokens[1:]):
            ll += logA[u, v]
        return ll

    def predict(self, corpus_int: List[List[int]]) -> List[int]:
        preds = []
        for sent in corpus_int:
            scores = [
                self._log_likelihood(sent, c) + self.logPriors[c]
                for c in range(self.n_classes)
            ]
            preds.append(int(np.argmax(scores)))
        return preds
