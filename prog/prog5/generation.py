# generation.py — version avec modèle auto‑téléchargeable
from __future__ import annotations
import random
import string
from typing import List

import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api


def load_default_model() -> KeyedVectors:
    """
    Charge un modèle léger d’embeddings (~15 Mo) utilisable sans fichier externe.
    """
    print("Chargement du modèle léger 'glove-wiki-gigaword-50'")
    return api.load("glove-wiki-gigaword-50")


def _clean(text: str) -> List[str]:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def _nearest_valid_word(vector: np.ndarray,
                        kv: KeyedVectors,
                        blacklist: set[str],
                        top_k: int = 10) -> str | None:
    for word, _ in kv.similar_by_vector(vector, topn=top_k):
        if word.isalpha() and word not in blacklist:
            return word
    return None


def generate_next_third(text: str,
                        kv: KeyedVectors,
                        context_size: int = 10,
                        random_sigma: float = 0.05) -> str:
    tokens = _clean(text)
    if not tokens:
        raise ValueError("Le texte d'entrée est vide ou mal formé.")

    gen_len = len(tokens)*4
    generated: list[str] = []
    blacklist = set(tokens)

    for _ in range(gen_len):
        ctx = [t for t in tokens[-context_size:] if t in kv]
        if not ctx:
            next_word = random.choice(kv.index_to_key[:3000])
        else:
            ctx_vec = np.mean([kv[t] for t in ctx], axis=0)
            ctx_vec += np.random.normal(scale=random_sigma, size=ctx_vec.shape)
            next_word = _nearest_valid_word(ctx_vec, kv, blacklist)
            if not next_word:
                next_word = random.choice(kv.index_to_key[:3000])

        tokens.append(next_word)
        generated.append(next_word)
        blacklist.add(next_word)

    return " ".join(generated)
