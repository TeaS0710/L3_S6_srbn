# main.py
import argparse
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import prepare_data, texts_to_int
from model import build_markov_models
from classifier import BigramNBClassifier
from generation import generate_next_third
from gensim.models import KeyedVectors
import gensim.downloader as api
from tqdm import tqdm

from openai_gen import gpt_extend_33


def read_file(path_str: str, parser) -> str:
    if not path_str:
        parser.error("--input requis")
    path = Path(path_str).resolve()
    if not path.exists():
        parser.error(f"Fichier non trouvé : {path}")
    return path.read_text(encoding="utf8")


def load_embeddings(kv_path: str | None) -> KeyedVectors:
    if kv_path:
        print(f"Chargement du modèle local : {kv_path}")
        return KeyedVectors.load(kv_path)

    model_name = "glove-wiki-gigaword-50"
    print(f"Téléchargement de '{model_name}' via Gensim")
    with tqdm(total=1, desc="Téléchargement embeddings") as pbar:
        model = api.load(model_name)
        pbar.update(1)
    return model


def mode_continue(text: str, kv_path: str | None) -> None:
    kv = load_embeddings(kv_path)
    continuation = generate_next_third(text, kv)
    print("Génération (~+33 %)")
    print(continuation)


def mode_continue_gpt(text: str, parser) -> None:
    print("Appel OpenAI pour prolonger le texte (~+33 %) …")
    try:
        continuation = gpt_extend_33(text)
    except RuntimeError as e:
        parser.error(str(e))
    print("Génération GPT (~+33 %)")
    print(continuation)


def mode_train_eval_predict(mode: str, input_text: str | None, parser) -> None:
    train_int, test_int, y_train, y_test, word2idx = prepare_data()
    V = len(word2idx)
    clf = BigramNBClassifier(*build_markov_models(train_int, y_train, V))

    if mode == "train":
        print("Modèle entraîné avec succès")
        print(f"Taille du vocabulaire : {V} tokens")
    elif mode == "eval":
        y_pred = clf.predict(test_int)
        print(classification_report(y_test, y_pred, target_names=["Poe", "Frost"]))
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
    elif mode == "predict":
        if not input_text:
            parser.error("--input requis en mode predict")
        user_text = [input_text.lower()]
        user_int = texts_to_int(user_text, word2idx)
        label = clf.predict(user_int)[0]
        author = "Edgar Allan Poe" if label == 0 else "Robert Frost"
        print(f"Ce texte est probablement écrit par : {author}")


def main():
    parser = argparse.ArgumentParser(
        description="Classificateur Poe / Frost basé sur des chaînes de Markov bigrammes."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "predict", "continue", "continue-gpt"],
        required=True,
        help="Choisit le mode d'exécution"
    )
    parser.add_argument("--input", help="Fichier texte à traiter")
    parser.add_argument("--kv-path", help="Fichier .kv à charger (mode continue)")

    args = parser.parse_args()

    # Lecture du texte uniquement si nécessaire
    text = read_file(args.input, parser) if args.mode in ["predict", "continue", "continue-gpt"] else None

    # Dispatch
    if args.mode == "continue":
        mode_continue(text, args.kv_path)
    elif args.mode == "continue-gpt":
        mode_continue_gpt(text, parser)
    else:
        mode_train_eval_predict(args.mode, text, parser)


if __name__ == "__main__":
    main()
