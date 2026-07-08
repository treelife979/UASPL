from pathlib import Path


def resolve_project_path(root_dir, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root_dir / path
