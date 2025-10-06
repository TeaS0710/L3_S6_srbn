#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application factorisée pour analyse statistique et clustering multilingue
depuis un corpus HTML hiérarchisé (dossiers/langues/sous-dossiers/fichiers.html).
Structure :
    - script_load_html.py        : chargement HTML -> data_loaded.json
    - script_preprocess.py       : prétraitement SpaCy -> preprocessed_LANG.json
    - script_stats.py            : stats -> stats.csv
    - script_viz.py              : visualisation
    - script_clustering.py       : clustering (2-3gram + 4-5gram)
    - main.py                    : orchestration
"""

import os
import sys
import json
import spacy

LANG_TO_MODEL = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm"
}

# script_preprocess.py

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_text(text, nlp):
    doc = nlp(text)
    total_tokens = sum(not t.is_space for t in doc)
    if total_tokens == 0:
        return [], 0.0
    nb_ent_tokens = sum(len(ent) for ent in doc.ents)
    proportion_ne = nb_ent_tokens / total_tokens
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_space and not token.is_punct and not token.is_stop and token.pos_ != "PROPN"
    ]
    return tokens, proportion_ne

def preprocess_language(data, lang_code, output_path):
    if lang_code not in LANG_TO_MODEL:
        raise ValueError(f"Langue non supportée: {lang_code}")
    nlp = spacy.load(LANG_TO_MODEL[lang_code])
    texts = data.get(lang_code, {})
    results = []
    for filename, text in texts.items():
        tokens, prop_ne = preprocess_text(text, nlp)
        results.append({
            "filename": filename,
            "lang": lang_code,
            "tokens": tokens,
            "prop_ne": prop_ne
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
