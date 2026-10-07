"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# try:
#     import some_third_party_module
# except ModuleNotFoundError as e:
#     print(f"[ERROR] Missing dependency: {e}")
#     input("\nPress Enter to exit...")

__version__ = "0.0.0"

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Code goes here")


def setup_logging(log_folder: Path = Path("Logs"), console_level: int = logging.DEBUG, enable_file_logging: bool = True, max_log_files: int | None = 30, file_level: int = logging.DEBUG, date_format: str = "%Y-%m-%dT%H:%M:%S", message_format: str = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s") -> Path | None:
    """Configures file and console logging and prunes old logs for this script."""
    if max_log_files is not None and max_log_files < 1:
        raise ValueError("max_log_files must be at least 1 or None.")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.propagate = False
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

        log_dir = Path(log_folder).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{timestamp}_{script_stem}.log"

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prune old logs specific to this script
        if max_log_files is not None and log_dir.exists():
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
    exit_code = 0
    try:
        setup_logging(max_log_files=5)
        # setup_logging(log_folder=Path(f"~/AppData/Local/Temp/{Path(__file__).stem}"), max_log_files=5) # C:\Users\YourName\AppData\Local\Temp\{script_name}
        main()

    except KeyboardInterrupt:
        print()
        logger.warning("Operation interrupted by user.")
        exit_code = 130

    except Exception as e:
        print()
        logger.exception("A fatal error has occurred: %s", e)
        exit_code = 1

    # input("Press Enter to exit...") # Uncomment to keep console open after script run
    sys.exit(exit_code)
