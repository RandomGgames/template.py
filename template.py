"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

from datetime import datetime
from pathlib import Path
import logging
import sys

__version__ = "0.0.0"

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Code goes here")


def setup_logging(log_folder: Path = Path("Logs"), console_level: int = logging.DEBUG, enable_file_logging: bool = True, max_log_files: int = 30, file_level: int = logging.DEBUG, date_format: str = "%Y-%m-%dT%H:%M:%S", message_format: str = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s") -> Path | None:
    """Configures file and console logging and prunes old logs for this script."""
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(message_format, datefmt=date_format)

    # Console Handler (always active)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path: Path | None = None

    # Optional File Handler
    if enable_file_logging:
        script_stem = Path(__file__).stem
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        log_dir = log_folder.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}_{script_stem}.log"

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prune old logs specific to this script
        if max_log_files > 0 and log_dir.exists():
            script_logs = sorted(
                [f for f in log_dir.glob("*.log") if f.name.endswith(f"_{script_stem}.log")],
                key=lambda p: p.stat().st_mtime,
            )
            while len(script_logs) > max_log_files:
                oldest = script_logs.pop(0)
                try:
                    oldest.unlink()
                except OSError:
                    pass

    return log_path


if __name__ == "__main__":
    PAUSE_ON_ERROR = True
    ALWAYS_PAUSE = False

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

    finally:
        # input("Press Enter to exit...")
        sys.exit(exit_code)
