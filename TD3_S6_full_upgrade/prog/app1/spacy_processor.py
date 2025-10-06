import spacy

# Assurez-vous d'avoir installé le modèle spaCy adéquat, par ex.:
#   python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_lg")

def lemmatize_tokens(tokens, use_pos=False):
    """
    Prend une liste de tokens (strings).
    Les passe à spaCy pour récupérer leur lemme (et la POS si use_pos=True).
    Retourne une liste de chaînes transformées.
    Ex si use_pos=False : ["maison", "école", ...]
    Ex si use_pos=True :  ["maison_NOUN", "école_NOUN", ...]
    """
    results = []

    # Pour optimiser, on crée un doc par token via nlp.pipe (plus rapide que nlp(token) en boucle)
    for doc in nlp.pipe(tokens):
        if len(doc) == 0:
            results.append("")
            continue

        token_spacy = doc[0]  # un seul token dans ce doc
        lemma = token_spacy.lemma_.lower()

        if use_pos:
            pos = token_spacy.pos_
            results.append(f"{lemma}_{pos}")
        else:
            results.append(lemma)

    return results
