# script_viz.py
import seaborn as sns

def plot_stats(csv_path, out_prefix):
    df = pd.read_csv(csv_path)
    df["text_id"] = df["filename"] + " (" + df["lang"] + ")"
    for metric in ["nb_tokens", "nb_types", "prop_lemmas", "prop_ne"]:
        plt.figure()
        sns.barplot(data=df, x="text_id", y=metric, hue="lang")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}.png")
        plt.close()
