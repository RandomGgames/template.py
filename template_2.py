"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

__version__ = "0.0.0"  # Major.Minor.Patch

logger = logging.getLogger(__name__)


def main():
    """Code goes here."""


class JsonArgsFilter(logging.Filter):
    """Format strings and Paths as JSON strings in log arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.args:
            return True

        args = record.args if isinstance(record.args, tuple) else (record.args,)
        processed = []

        for value in args:
            if isinstance(value, Path):
                processed.append(json.dumps(value.as_posix()))
            elif isinstance(value, str):
                processed.append(json.dumps(value))
            elif isinstance(value, (int, float, bool)) or value is None:
                processed.append(value)
            else:
                processed.append(json.dumps(value, default=str))

        record.args = tuple(processed)
        return True


def enforce_max_log_count(log_dir: Path, max_log_files: int = 30) -> None:
    """Remove old log files when the maximum log count is exceeded."""
    if max_log_files <= 0:
        return

    script_name = Path(__file__).stem
    log_files = sorted(log_dir.glob(f"*__{script_name}.log"))

    for log_file in log_files[:-max_log_files]:
        try:
            log_file.unlink()
        except OSError as e:
            logger.debug("Failed removing old log %s: %s", log_file, e)


def setup_logging() -> Path:
    """Set up console and per-run file logging and return the log path."""
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addFilter(JsonArgsFilter())

    log_dir = Path(__file__).resolve().parent / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).stem
    log_path = log_dir / f"{timestamp}__{script_name}.log"

    formatter = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S",)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    enforce_max_log_count(log_dir, max_log_files=5)

    return log_path


def bootstrap() -> int:
    """Set up the script, run main(), and handle unexpected errors."""
    exit_code = 0

    try:
        setup_logging()
        main()
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:
        logger.exception("A fatal error has occurred: %s", e)
        exit_code = 1

    if exit_code != 0:
        input("Press Enter to exit...")
    # input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
