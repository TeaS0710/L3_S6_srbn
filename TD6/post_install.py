# post_requirements.py
import gensim.downloader as api
MODEL = "glove-wiki-gigaword-50"

print(f"Téléchargement automatique du modèle « {MODEL} »…")
api.load(MODEL)
print("OK")
