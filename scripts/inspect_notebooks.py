import os
from pathlib import Path
import json

downloads = Path(os.environ.get("USERPROFILE", "")) / "Downloads"
nb1 = downloads / "Thin Images Ghana" / "Thin Images With Annotations" / "Copy of Untitled0.ipynb"
nb2 = downloads / "Malaria_Classification_from_smears.ipynb"

def inspect_notebook(nb_path):
    if not nb_path.exists():
        print(f"Notebook {nb_path} does not exist.")
        return
    print(f"\n=== Inspecting Notebook: {nb_path.name} ===")
    try:
        with open(nb_path, "r", encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        print(f"Total cells: {len(cells)}")
        for idx, cell in enumerate(cells[:8]):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", []))
            if source.strip():
                print(f"--- Cell {idx} ({cell_type}) ---")
                print(source[:300].strip())
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")

inspect_notebook(nb1)
inspect_notebook(nb2)
