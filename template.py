"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import datetime
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

logger = logging.getLogger(__name__)
log_buffer = logging.handlers.MemoryHandler(
    capacity=0,
    flushLevel=logging.CRITICAL,
    target=None,
)
logger.addHandler(log_buffer)
logger.setLevel(logging.DEBUG)


class ScriptSettings:
    def __init__(self):
        """Place any code for whatever script is being writen here"""


class LogSettings:
    def __init__(self):
        self.mode: typing.Literal["per_run", "latest", "per_day", "single_file", "console_only"] = "per_run"
        self.folder: Path = Path(r"Logs")
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
        self.ScriptSettings = ScriptSettings()
        self.LogSettings = LogSettings()
        self.RuntimeSettings = RuntimeSettings()


def main(config: Config):
    """Code goes here"""


def read_json_file(file_path: Path) -> dict | list | None:
    """
    Safely reads and parses a JSON file.
    """
    if not file_path.exists():
        logger.warning("File not found: %s", json.dumps(str(file_path)))
        raise FileNotFoundError("File not found")

    try:
        data = json.loads(file_path.read_text(encoding='utf-8'))
        logger.info("Successfully read data from %s", json.dumps(str(file_path)))
        return data

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format in %s: %s", json.dumps(str(file_path)), e)
        return None

    except Exception as e:
        logger.error("Unexpected error reading %s: %s", json.dumps(str(file_path)), e)
        return None


def write_json_file(file_path: Path, data: dict | list) -> bool:
    """
    Writes data to a JSON file atomically.
    """
    file_path = Path(file_path).absolute()

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Created %s", json.dumps(str(file_path.parent.as_posix())))

    temp_file_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=str(file_path.parent), encoding='utf-8', suffix=".tmp", delete=False) as tf:
            # Get file path from tempfile object
            temp_file_path = Path(tf.name)
            json.dump(data, tf, indent=4)
            tf.flush()
            os.fsync(tf.fileno())

        # Atomic swap
        temp_file_path.replace(file_path)
        logger.info("Successfully saved to %s", json.dumps(str(file_path)))
        return True

    except (KeyboardInterrupt, SystemExit):
        logger.error("Write interrupted for %s. Cleaning up.", json.dumps(str(file_path)))
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        raise

    except Exception as e:
        logger.error("Failed to write to %s: %s", json.dumps(str(file_path)), e)
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
        return False


def enforce_max_log_count(dir_path: Path, max_count: int, script_name: str) -> None:
    """
    Enforce a maximum number of log files for this script.
    Deletes the oldest logs based on filename ordering.

    Rules:
    - Only affects files ending with `.log`
    - Only affects logs that contain the script_name
    - Sorting is done by filename (lexicographically)
    """
    if max_count <= 0:
        return
    if not dir_path.exists():
        return
    log_files = [
        f for f in dir_path.glob("*.log")
        if script_name in f.name
    ]
    if len(log_files) <= max_count:
        return
    log_files.sort(key=lambda p: p.name)
    to_delete = log_files[:-max_count]
    for file in to_delete:
        try:
            file.unlink()
            logger.debug("Removed %s", json.dumps(file.absolute().as_posix()))
        except Exception:
            # Avoid raising during bootstrap
            pass


def setup_logging(logger_obj: logging.Logger, log_settings: LogSettings, config: Config) -> Path | None:
    """Set up file and console logging with flexible modes and rotation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    day_stamp = datetime.now().strftime("%Y%m%d")
    script_name = Path(__file__).stem
    pc_name = socket.gethostname()

    log_path: Path | None = None

    if log_settings.mode != "console_only":
        log_dir = (log_settings.folder if isinstance(log_settings.folder, Path) else Path(log_settings.folder))
        log_dir = log_dir.expanduser().resolve()
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            logger_obj.debug("Created log folder: %s", log_dir.as_posix())

        match log_settings.mode:
            case "per_run":
                log_path = log_dir / f"{timestamp}__{script_name}__{pc_name}.log"
            case "latest":
                log_path = log_dir / f"latest_{script_name}__{pc_name}.log"
            case "per_day":
                log_path = log_dir / f"{day_stamp}__{script_name}__{pc_name}.log"
            case "single_file":
                log_path = log_dir / f"{script_name}__{pc_name}.log"
            case _:
                log_path = log_dir / f"{timestamp}__{script_name}__{pc_name}.log"

    logger_obj.handlers.clear()
    logger_obj.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        log_settings.message_format,
        datefmt=log_settings.date_format,
    )

    # File handler
    file_handler: logging.Handler | None = None
    if log_path:
        match log_settings.mode:
            case "per_day":
                file_handler = TimedRotatingFileHandler(
                    filename=log_path,
                    when="midnight",
                    interval=1,
                    backupCount=log_settings.max_files or 0,
                    encoding="utf-8",
                )
            case "single_file" | "latest" | "per_run":
                file_mode = "a" if log_settings.mode == "single_file" else "w"
                file_handler = logging.FileHandler(
                    log_path,
                    mode=file_mode,
                    encoding="utf-8",
                )

    if file_handler:
        file_handler.setLevel(log_settings.file_level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_settings.console_level)
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)

    # Write the banner after handlers are attached
    write_banner(logger_obj, config)

    # Flush logs buffer from prior to logging initialization
    if "log_buffer" in globals() and 'log_buffer' in globals():
        class _ForwardToLogger(logging.Handler):
            def emit(self, record):
                logger_obj.handle(record)

        forward_handler = _ForwardToLogger()
        log_buffer.setTarget(forward_handler)
        log_buffer.flush()
        log_buffer.close()

    # Enforce max log count
    if log_settings.max_files and log_path and log_settings.mode not in ("per_day", "console_only"):
        try:
            enforce_max_log_count(
                dir_path=log_path.parent,
                max_count=log_settings.max_files,
                script_name=script_name,
            )
        except Exception as e:
            logger_obj.debug("Log pruning skipped: %s", e)

    return log_path


def write_banner(logger_obj: logging.Logger, config: Config):
    """Writes a clean, unformatted session header directly to the output streams."""
    script_name = Path(__file__).name
    separator = "-" * 80
    header_text = (
        f"{separator}\n"
        f"SCRIPT          | {json.dumps(str(script_name))}\n"
        f"VERSION         | {__version__}\n"
        f"USER/HOST       | {os.getlogin()} on {socket.gethostname()}\n"
        f"EXECUTION START | {datetime.now().isoformat(sep='T', timespec='milliseconds')}\n"
        f"DIRECTORY       | {json.dumps(str(Path.cwd().as_posix()))}\n"
        f"PLATFORM        | {platform.system()} {platform.release()}\n"
        f"RUNTIME         | Python {sys.version.split()[0]}\n"
        f"{separator}\n"
    )

    for handler in logger_obj.handlers:
        if isinstance(handler, logging.StreamHandler):
            try:
                handler.stream.write(header_text)
                handler.flush()
            except Exception:
                continue


def to_class_str(obj, key_name=None):
    """
    Recursively converts objects to a single-line Pythonic string.
    """
    if isinstance(obj, Path):
        val_str = f"Path({repr(str(obj))})"
    elif isinstance(obj, list):
        items = [to_class_str(item) for item in obj]
        val_str = f"[{', '.join(items)}]"
    elif hasattr(obj, "__dict__"):
        class_name = type(obj).__name__
        attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}

        if not attrs:
            val_str = f"{class_name}()"
        else:
            # Build "key=value" pairs, but skip "key=" if key matches ClassName
            parts = [to_class_str(v, k) for k, v in attrs.items()]
            val_str = f"{class_name}({', '.join(parts)})"
    else:
        val_str = repr(obj)

    # Omit "key=" prefix if key name matches the ClassName
    if key_name is None or key_name == type(obj).__name__:
        return val_str

    return f"{key_name}={val_str}"


def bootstrap():
    exit_code = 0
    log_path = None

    config = Config()

    try:
        log_path = setup_logging(logger_obj=logger, log_settings=config.LogSettings, config=config)

        config_summary = json.dumps(to_class_str(config))
        logger.info("Configuration loaded: %s", config_summary)

        main(config)

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:
        logger.exception("A fatal error has occurred: %s", e)
        exit_code = 1
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if config.LogSettings.open_log_after_run and log_path and log_path.exists():
        try:
            match sys.platform:
                case plat if plat.startswith("win"):  # Windows
                    os.startfile(log_path)
                case "darwin":  # macOS
                    os.system(f'open "{log_path}"')
                case _:  # Linux / others
                    os.system(f'xdg-open "{log_path}"')
        except Exception as e:
            logger.warning("Failed to open log file: %s", e)

    if config.RuntimeSettings.always_pause or (config.RuntimeSettings.pause_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
