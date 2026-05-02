from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import load_settings
from research_assistant.vectorstore import recreate_vectorstore


def main() -> None:
    settings = load_settings()
    recreate_vectorstore(settings)
    print(f"Indexed corpus into {settings.qdrant_collection}")


if __name__ == "__main__":
    main()
