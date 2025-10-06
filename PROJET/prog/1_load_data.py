import os
import json
from bs4 import BeautifulSoup

def load_html_from_folder(folder_path):
    data = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".html"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")

                for script_tag in soup(["script", "style"]):
                    script_tag.decompose()

                # Récupération du texte brut depuis l’HTML
                text_content = soup.get_text(separator=" ")

                text_content = " ".join(text_content.split())

            data[file_name] = text_content

    return data

def main():
    base_dir = "../data/corpus-multi"  # Adapter selon votre arborescence
    languages = os.listdir(base_dir)
    all_data = {}

    for lang in languages:
        lang_path = os.path.join(base_dir, lang)
        if os.path.isdir(lang_path):
           texts = load_html_from_folder(lang_path)
            all_data[lang] = texts

    # Sauvegarde au format JSON
    output_file = "../outputs/data_loaded.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
