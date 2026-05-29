# ============================================================
# Generate Folder Structure Report for Experiments Directory
# ============================================================

from pathlib import Path

ROOT_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")

OUTPUT_FILE = ROOT_DIR / "experiments_folder_structure.txt"


def generate_tree(folder, prefix=""):
    lines = []

    items = sorted(
        folder.iterdir(),
        key=lambda x: (x.is_file(), x.name.lower())
    )

    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "

        lines.append(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if i == len(items) - 1 else "│   "
            lines.extend(generate_tree(item, prefix + extension))

    return lines


def main():
    print("=" * 70)
    print("Generating experiments folder structure...")
    print("=" * 70)

    tree_lines = [ROOT_DIR.name]
    tree_lines.extend(generate_tree(ROOT_DIR))

    OUTPUT_FILE.write_text(
        "\n".join(tree_lines),
        encoding="utf-8"
    )

    print(f"Folder structure saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()