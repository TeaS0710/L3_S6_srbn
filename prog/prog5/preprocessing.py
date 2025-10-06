#PREPROCESSING
from pathlib import Path
import string
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

PUNCT_TRANS = str.maketrans("", "", string.punctuation)
UNK_TOKEN = "<unk>"

def load_corpora(files: List[Path]) -> Tuple[List[str], List[int]]:

    input_texts, labels = [], []
    for label, f in enumerate(files):
        with f.open(encoding="utf8") as fh:
            for line in fh:
                clean = (
                    line.lower()
                    .rstrip()
                    .translate(PUNCT_TRANS)
                )
                if clean:
                    input_texts.append(clean)
                    labels.append(label)
    return input_texts, labels

def build_vocab(train_texts: List[str]) -> Dict[str, int]:

    word2idx = {UNK_TOKEN: 0}
    idx = 1
    for line in train_texts:
        for token in line.split():
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1
    return word2idx

def texts_to_int(texts: List[str], word2idx: Dict[str, int]) -> List[List[int]]:

    unk = word2idx[UNK_TOKEN]
    corpus_int = []
    for line in texts:
        corpus_int.append([word2idx.get(tok, unk) for tok in line.split()])
    return corpus_int

def prepare_data(
    poe_path: str = "data/edgar_allan_poe.txt",
    frost_path: str = "data/robert_frost.txt",
    test_size: float = 0.3,
    random_state: int = 42,
):

    files = [Path(poe_path), Path(frost_path)]
    texts, labels = load_corpora(files)

    train_text, test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    word2idx = build_vocab(train_text)
    train_int = texts_to_int(train_text, word2idx)
    test_int = texts_to_int(test_text, word2idx)
    return train_int, test_int, y_train, y_test, word2idx
