from pathlib import Path

"""
[paths.py]
You should add folder you are going to reference alot here usually,
Added auto resolver for if we want to load from configs
"""

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"


def resolve_path(path: str) -> tuple[Path, bool]:
    p = ROOT / path

    # Return a bool to check if file actually exists or not
    exists = p.is_file()
    return p, exists
