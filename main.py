import argparse
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


def main() -> int:
    args = parse_args()
    data = load_data(args.data_folder)

    for id, content in data.items():
        print(f"{id} | {textwrap.shorten(content, width=60)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
