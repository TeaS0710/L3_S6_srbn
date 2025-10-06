import os
import json
import matplotlib.pyplot as plt

def load_processed_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_stat_per_lang(data, stat_key, ylabel, title):
    langs = []
    values = []

    for lang, docs in data.items():
        values_lang = [doc[stat_key] for doc in docs if stat_key in doc]
        if not values_lang:
            continue
        langs.append(lang)
        values.append(values_lang)

    # Plot : Boxplot pour chaque langue
    plt.figure(figsize=(10, 6))
    plt.boxplot(values, labels=langs)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Langues")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    data_path = "../pipeline_results/processed_multilang.json"
    data = load_processed_data(data_path)

    plot_stat_per_lang(data, "n_tokens", "Nombre de tokens", "Distribution du nombre de tokens par texte")
    plot_stat_per_lang(data, "n_types", "Nombre de types", "Distribution du vocabulaire par texte")
    plot_stat_per_lang(data, "prop_lemmes", "Proportion de lemmes", "Proportion de lemmes par texte")
    plot_stat_per_lang(data, "prop_propn", "Proportion de noms propres", "Proportion de noms propres par texte")

if __name__ == "__main__":
    main()
