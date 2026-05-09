"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import json
import logging
import logging.handlers
import os
import platform
import socket
import sys
import tempfile
import typing
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)

__version__ = "0.0.0"  # Major.Minor.Patch

log_buffer = logging.handlers.MemoryHandler(
    capacity=0,
    flushLevel=logging.CRITICAL,
    target=None,
)

logger.addHandler(log_buffer)
logger.setLevel(logging.DEBUG)


class ScriptSettings:
    def __init__(self):
        """Place any code for whatever script is being written here."""


class LogSettings:
    def __init__(self):
        self.mode: typing.Literal["per_run", "latest", "per_day", "single_file", "console_only"] = "per_run"
        self.folder: Path = Path("Logs")
        self.console_level: int = logging.DEBUG
        self.file_level: int = logging.DEBUG
        self.date_format: str = "%Y-%m-%dT%H:%M:%S"
        self.message_format: str = "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(module)s:%(funcName)s - %(message)s"
        self.max_files: int | None = 30
        self.open_log_after_run: bool = False


class RuntimeSettings:
    def __init__(self):
        self.pause_on_error: bool = True
        self.always_pause: bool = False


class Config:
    def __init__(self):
        self.script_settings = ScriptSettings()
        self.log_settings = LogSettings()
        self.runtime_settings = RuntimeSettings()


def main(config: Config):
    """Code goes here"""


def j(value) -> str:
    """
    Converts values into compact JSON-formatted strings for logging.

    Special handling:
    - Path -> POSIX string path
    - Other objects -> normal json serialization fallback to str()
    """
    if isinstance(value, Path):
        value = value.as_posix()

    return json.dumps(value, default=str)


def read_json_file(file_path: Path) -> dict | list | None:
    """
    Safely reads and parses a JSON file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning("File not found: %s", j(file_path))
        raise FileNotFoundError(file_path)

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", j(file_path), e)
        return None

    except OSError as e:
        logger.error("Failed reading %s: %s", j(file_path), e)
        return None

    logger.info("Successfully read %s", j(file_path))
    return data


def write_json_file(file_path: Path, data: dict | list) -> bool:
    """
    Writes data to a JSON file atomically.
    """
    file_path = Path(file_path).absolute()

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    except OSError as e:
        logger.error("Failed creating directory %s: %s", j(file_path.parent), e)
        return False

    temp_file_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=str(file_path.parent), encoding="utf-8", suffix=".tmp", delete=False) as tf:
            temp_file_path = Path(tf.name)

            json.dump(data, tf, indent=4)

            tf.flush()
            os.fsync(tf.fileno())

        temp_file_path.replace(file_path)

        logger.info("Successfully saved to %s", j(file_path))
        return True

    except (KeyboardInterrupt, SystemExit):
        logger.error("Write interrupted for %s. Cleaning up.", j(file_path))

        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

        raise

    except OSError as e:
        logger.error("Filesystem error writing %s: %s", j(file_path), e)

    except Exception as e:
        logger.error("Unexpected error writing %s: %s", j(file_path), e)

    if temp_file_path and temp_file_path.exists():
        temp_file_path.unlink(missing_ok=True)

    return False


def enforce_max_log_count(dir_path: Path, max_count: int, script_name: str) -> None:
    """
    Enforce a maximum number of log files for this script.

    Rules:
    - Only affects files ending with `.log`
    - Only affects logs that contain the script name
    - Sorting is performed lexicographically by filename
    """
    if max_count <= 0:
        return

    if not dir_path.exists():
        return

    log_files = [f for f in dir_path.glob("*.log") if script_name in f.name]

    if len(log_files) <= max_count:
        return

    log_files.sort(key=lambda p: p.name)

    to_delete = log_files[:-max_count]

    for file in to_delete:
        try:
            file.unlink()

            logger.debug("Removed old log %s", j(file))

        except OSError as e:
            logger.debug("Failed removing old log %s: %s", j(file), e)


def build_log_path(log_settings: LogSettings) -> Path | None:
    """
    Builds the final log file path based on logging mode.
    """
    if log_settings.mode == "console_only":
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    day_stamp = datetime.now().strftime("%Y%m%d")

    script_name = Path(__file__).stem
    pc_name = socket.gethostname()

    log_dir = Path(log_settings.folder).expanduser().resolve()

    match log_settings.mode:
        case "per_run":
            filename = f"{timestamp}__{script_name}__{pc_name}.log"
        case "latest":
            filename = f"latest_{script_name}__{pc_name}.log"
        case "per_day":
            filename = f"{day_stamp}__{script_name}__{pc_name}.log"
        case "single_file":
            filename = f"{script_name}__{pc_name}.log"
        case _:
            filename = f"{timestamp}__{script_name}__{pc_name}.log"

    return log_dir / filename


def setup_logging(logger_obj: logging.Logger, log_settings: LogSettings) -> Path | None:
    """
    Set up console and file logging.
    """
    logger_obj.handlers.clear()

    logger_obj.setLevel(logging.DEBUG)

    logger_obj.propagate = False

    log_path = build_log_path(log_settings)

    formatter = logging.Formatter(
        log_settings.message_format,
        datefmt=log_settings.date_format,
    )

    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        except OSError as e:
            raise RuntimeError(f"Failed creating log directory "    f"{log_path.parent}") from e

        file_handler: logging.Handler

        match log_settings.mode:
            case "per_day":
                file_handler = TimedRotatingFileHandler(filename=log_path, when="midnight", interval=1, backupCount=log_settings.max_files or 0, encoding="utf-8")
            case "single_file":
                file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            case _:
                file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

        file_handler.setLevel(log_settings.file_level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_settings.console_level)
    console_handler.setFormatter(formatter)

    logger_obj.addHandler(console_handler)

    write_banner(logger_obj)

    if log_buffer:
        class _ForwardToLogger(logging.Handler):
            def emit(self, record):
                logger_obj.handle(record)

        forward_handler = _ForwardToLogger()
        log_buffer.setTarget(forward_handler)
        log_buffer.flush()
        log_buffer.close()

    if (log_settings.max_files and log_path and log_settings.mode not in ("per_day", "console_only")):
        enforce_max_log_count(dir_path=log_path.parent, max_count=log_settings.max_files, script_name=Path(__file__).stem)

    return log_path


def write_banner(logger_obj: logging.Logger):
    """
    Writes a clean session banner without log prefixes.
    """
    separator = "-" * 80

    banner = (
        f"{separator}\n"
        f"SCRIPT          | {j(Path(__file__).name)}\n"
        f"VERSION         | {__version__}\n"
        f"USER/HOST       | {os.getlogin()} "
        f"on {socket.gethostname()}\n"
        f"EXECUTION START | "
        f"{datetime.now().isoformat(timespec='milliseconds')}\n"
        f"DIRECTORY       | {j(Path.cwd())}\n"
        f"PLATFORM        | "
        f"{platform.system()} {platform.release()}\n"
        f"RUNTIME         | Python {sys.version.split()[0]}\n"
        f"{separator}"
    )

    original_formatters = {}

    class RawFormatter(logging.Formatter):
        """
        Formatter that outputs only the log message with no prefixes.
        """

        def format(self, record):
            return record.getMessage()

    try:
        for handler in logger_obj.handlers:
            original_formatters[handler] = handler.formatter
            handler.setFormatter(RawFormatter())

        logger_obj.info(banner)

    finally:
        for handler, formatter in original_formatters.items():
            handler.setFormatter(formatter)


def bootstrap():
    exit_code = 0

    log_path: Path | None = None

    config = Config()

    try:
        log_path = setup_logging(logger_obj=logger, log_settings=config.log_settings)

        main(config)

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")

        exit_code = 130

    except Exception as e:
        logger.exception("A fatal error has occurred: %s", e)

        exit_code = 1

    if (config.log_settings.open_log_after_run and log_path and log_path.exists()):
        try:
            match sys.platform:
                case plat if plat.startswith("win"):
                    os.startfile(log_path)
                case "darwin":
                    os.system(f'open "{log_path}"')
                case _:
                    os.system(f'xdg-open "{log_path}"')

        except Exception as e:
            logger.warning("Failed to open log file: %s", e)

    if (config.runtime_settings.always_pause or (config.runtime_settings.pause_on_error and exit_code != 0)):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
