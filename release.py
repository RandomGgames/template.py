"""
Python Project Release Builder

Creates a versioned ZIP archive for a Python project
based on a TOML configuration file. Missing files are
warned about and skipped.
"""

import json
import re
import sys
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path


def zip_files(files: list[Path], zip_path: str | Path, compresslevel: int = 6) -> None:
    """Create a ZIP archive from the given list of files."""
    files = [Path(f) for f in files]
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zipf:
        for path in files:
            zipf.write(path, arcname=path.name)

    print(f"Created release archive: {json.dumps(str(zip_path))}")


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, "_", name).strip()
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.replace(" ", "_")
    return sanitized or "file"


def read_toml(file_path: str | Path) -> dict:
    """Read a TOML file and return it as a dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"TOML config not found: {json.dumps(str(path))}")
    with path.open("rb") as f:
        return tomllib.load(f)


def main():
    # Locate config file next to script
    script_path = Path(__file__)
    config_path = script_path.with_name(f"{script_path.stem}_config.toml")
    config = read_toml(config_path)

    name = str(config.get("name", "project"))
    version = str(config.get("version", "0.0.0")).lstrip("v")
    config_files = config.get("files", [])

    if not isinstance(config_files, list) or not all(isinstance(f, str) for f in config_files):
        raise TypeError("Expected list of strings for key 'files' in config")

    files = [Path(f) for f in config_files]

    # --- File cleanup / validation ---
    valid_files = []
    for file in files:
        if file.is_file():
            valid_files.append(file)
        else:
            print(f"Warning: File not found, skipping: {json.dumps(str(file))}")

    if not valid_files:
        print("No valid files found to include in ZIP. Exiting.")
        sys.exit(1)

    # --- Generate ZIP ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{name}_v{version}_{timestamp}.zip"
    zip_name = sanitize_filename(zip_name)

    zip_files(valid_files, zip_name)


if __name__ == "__main__":
    sys.exit(main())
