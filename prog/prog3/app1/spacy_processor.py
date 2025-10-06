import spacy

SPACY_MODELS = {
    #"fr": "fr_core_news_sm",
    #"en": "en_core_web_sm",
    #"de": "de_core_news_sm",
    #"es": "es_core_web_sm",
    #"fi": "fi_core_news_sm",
    #"sl": "sl_core_news_trf",  # ou sl_core_news_sm
    #"da": "da_core_news_trf",  # ou da_core_news_sm
    "sl": "sl_core_news_sm",
    "da": "da_core_news_sm",
}

FALLBACK_MODEL = "xx_sent_ud_sm"
loaded_models = {}

def get_nlp(lang_code):
    """
    Charge le modèle spaCy adapté à la langue.
    Si non supportée, utilise le modèle multilingue xx_sent_ud_sm.
    """
    if lang_code in loaded_models:
        return loaded_models[lang_code]

    model_name = SPACY_MODELS.get(lang_code, FALLBACK_MODEL)

    try:
        nlp = spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"[ERREUR] Le modèle spaCy '{model_name}' est manquant.\n"
            f"→ Exécute : python -m spacy download {model_name}"
        )

    loaded_models[lang_code] = nlp
    return nlp


def tokenize_by_space(text):
    """
    Tokenisation naïve par espaces (sans traitement linguistique).
    """
    return text.strip().split()


def lemmatize_tokens(tokens, use_pos=False, lang="xx"):
    """
    Lemmatisation d'une liste de tokens avec spaCy.
    """
    nlp = get_nlp(lang)
    results = []

    for doc in nlp.pipe(tokens):
        if len(doc) == 0:
            results.append("")
            continue

        token_spacy = doc[0]
        lemma = token_spacy.lemma_.lower()
        if use_pos:
            results.append(f"{lemma}_{token_spacy.pos_}")
        else:
            results.append(lemma)

    return results


def process_texts_by_lang(corpus_dict, use_pos=True):
    """
    Applique tokenisation, filtrage linguistique, NER et lemmatisation à un corpus multilingue.
    Retourne : {langue: [ {tokens, labels, lemmes} ]}
    """
    results_by_lang = {}

    for lang, texts in corpus_dict.items():
        print(f"[INFO] Traitement spaCy pour '{lang}'")
        nlp = get_nlp(lang)
        processed_docs = []

        for text in texts:
            doc = nlp(text)

            # Étape 1 : extraction de tous les tokens
            tokens = []
            labels = []
            lemmes = []

            for tok in doc:
                if tok.pos_ == "PROPN" or tok.is_stop:
                    continue  # on ignore les noms propres et les stopwords

                tokens.append(tok.text)
                labels.append(f"{tok.ent_iob_}-{tok.ent_type_}" if tok.ent_type_ else "O")

                if use_pos:
                    lemmes.append(f"{tok.lemma_.lower()}_{tok.pos_}")
                else:
                    lemmes.append(tok.lemma_.lower())

            processed_docs.append({
                "tokens": tokens,
                "labels": labels,
                "lemmes": lemmes,
            })

        results_by_lang[lang] = processed_docs

        print(f"[INFO] → {len(texts)} textes à traiter pour la langue '{lang}'")
        for i, doc in enumerate(nlp.pipe(texts, batch_size=32), 1):
            print(f"\r[spaCy] Traitement {i}/{len(texts)} pour '{lang}'", end="", flush=True)


    return results_by_lang
