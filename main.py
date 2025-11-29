import argparse
import string
import sys
import textwrap
from pathlib import Path


def load_data(data_folder: Path) -> dict[int, str]:
    data_files = data_folder.glob("*.txt")
    data_dict = {}
    for file_path in data_files:
        file_id = int(file_path.stem)
        file_data = file_path.read_text()
        data_dict[file_id] = file_data

    return data_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process some data files.")
    parser.add_argument(
        "--data-folder",
        type=Path,
        required=False,
        default=Path("./data"),
        help="Path to the folder containing data files. Default: %(default)s",
    )
    return parser.parse_args()


# TASK 1: Build a simple keyword search function
# TASK 2: Add "scoring" to the keyword search function
# TASK 3: Create an inverted index
# TASK 4: Create a TF-IDF index
# TASK 5: Implement semantic search using embeddings


def tokenize(text: str) -> list[str]:
    """
    TODO
    """

    text = text.lower().strip()
    tokens = text.split()

    final_tokens = [tok for tok in tokens if tok not in string.punctuation]

    # final_tokens = []
    # for tok in tokens:
    #     if tok in string.punctuation:
    #         continue
    #     final_tokens.append(tok)

    return final_tokens


def build_token_index(data: dict[int, str]) -> dict[int, list[str]]:
    """
    TODO: Build a token index for the given data.
    Return a mapping:
        document_id -> ["list", "of", "tokens", "in", "the", "document"]
    """


def keyword_search(data, keywords: list[str]) -> dict[int, str]:
    """
    TASK: Make the function use "AND" logic instead of "OR" logic.
    """

    matching_doc_ids = set()

    for document_id, content in data.items():
        all_keywords_found = True
        for keyword in keywords:
            if keyword not in content:
                all_keywords_found = False

        if all_keywords_found:
            matching_doc_ids.add(document_id)

    return {doc_id: data[doc_id] for doc_id in matching_doc_ids}


def main() -> int:
    args = parse_args()
    data = load_data(args.data_folder)

    results = keyword_search(data, ["machine", "learning"])

    for id, content in results.items():
        print(f"{id} | {textwrap.shorten(content, width=400)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
