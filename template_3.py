"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


__version__ = "0.0.0"  # Major.Minor.Patch

logger = logging.getLogger(__name__)


def enforce_max_log_count(log_dir: Path, max_count: int):
    if max_count <= 0:
        return

    script_name = Path(__file__).stem
    log_files = sorted(log_dir.glob(f"*__{script_name}.log"))

    for log_file in log_files[:-max_count]:
        try:
            log_file.unlink()
        except OSError:
            pass


def setup_logging(max_log_files: int):
    log_dir = Path(__file__).resolve().parent / "Logs"
    log_dir.mkdir(exist_ok=True)

    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}__{script_name}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    logger.setLevel(logging.DEBUG)

    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    enforce_max_log_count(log_dir, max_log_files)


def main():
    """Code goes here."""


def bootstrap():
    exit_code = 0

    try:
        setup_logging(max_log_files=30)
        main()
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception:
        logger.exception("A fatal error has occurred.")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
