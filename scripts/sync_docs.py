"""Copy the sampsizeval package into docs/py/ so the Pyodide web demo uses the
same code as the installed package. Run after editing sampsizeval/.

    python scripts/sync_docs.py
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampsizeval"
DST = ROOT / "docs" / "py" / "sampsizeval"

MODULES = ["__init__.py", "development.py", "validation_closed.py",
           "validation_sim.py", "compare_auc.py", "data.py"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        shutil.copy2(SRC / name, DST / name)
        print(f"copied {name}")
    print(f"\nSynced {len(MODULES)} modules -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
