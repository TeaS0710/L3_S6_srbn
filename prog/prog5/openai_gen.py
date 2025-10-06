# openai_gen.py
import math
import tiktoken
from openai import OpenAI
from tqdm import tqdm

# Clé API en dur
client = OpenAI(api_key="sk- METTRE VOTRE CLEF API ICI")

MODEL = "gpt-4o-mini"

def _estimate_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model(MODEL)
    return len(enc.encode(text))

def gpt_extend_33(text: str) -> str:
    tokens_src = _estimate_tokens(text)
    target_tokens = math.ceil(tokens_src * 0.33)

    prompt = (
        "Tu es un assistant littéraire. "
        "Prolonge le texte suivant en gardant le style, la voix et le registre, "
        f"jusqu’à ajouter environ {target_tokens} tokens supplémentaires "
        "(≈33 % de longueur en plus). Ne répète pas le texte source, continue‑le."
    )

    with tqdm(total=1, desc="OpenAI") as pbar:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": text},
            ],
            max_tokens=target_tokens,
            temperature=0.8,
        )
        pbar.update(1)

    return resp.choices[0].message.content.strip()
