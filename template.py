"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import json
import logging
import pathlib
import socket
import sys
import time
# import traceback
from datetime import datetime
from typing import NamedTuple

import toml

logger = logging.getLogger(__name__)

__version__ = "0.0.0"  # Major.Minor.Patch


def load_cache(path: pathlib.Path | str = "cache.json") -> dict:
    """
    Loads a cache from the given path.

    Args:
    path (typing.Union[pathlib.Path, str], optional): The path of the cache file to load. Defaults to "cache.json".

    Returns:
    dict: The loaded cache.
    """
    logger.debug("Loading cache...")
    path = pathlib.Path(path)
    if path.exists():
        try:
            logger.debug("Reading cache file...")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug("Read cache file.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load cache from {json.dumps(str(path))}", e)
            data = {}
    else:
        logger.debug(f"Cache file {json.dumps(str(path))} does not exist. Generating blank cache...")
        data = {}

    logger.debug("Cache loaded.")
    return data


def validate_cache(data: dict) -> None:
    """
    Validates the given cache data.

    Args:
    cache (dict): The cache data to validate.
    """
    logger.debug("Validating cache...")
    for file_path in list(data["files"].keys()):
        if not pathlib.Path(file_path).exists():
            logger.debug(f"Removing non-existent file {json.dumps(str(file_path))} from cache.")
            del data["files"][file_path]
    logger.debug("Cache validated successfully.")


def save_cache(data: dict, path: pathlib.Path | str = "cache.json") -> None:
    """
    Saves the given cache data to the given path.

    Args:
    data (dict): The cache data to save.
    path (typing.Union[pathlib.Path, str], optional): The path of the cache file to save. Defaults to "cache.json".
    """
    logger.debug("Saving cache...")
    path = pathlib.Path(path)
    try:
        cache_dir = path.parent
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
            logger.debug(f"Created cache directory {json.dumps(str(cache_dir))}.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            logger.debug("Saved cache.")
    except Exception as e:
        logger.error(f"Failed to save cache to {json.dumps(str(path))}", e)
        raise


def read_toml(file_path: pathlib.Path | str) -> dict:
    """
    Reads a TOML file and returns its contents as a dictionary.

    Args:
    file_path (pathlib.Path | str): The file path of the TOML file to read.

    Returns:
    dict: The contents of the TOML file as a dictionary.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {json.dumps(str(file_path))}")
    data = toml.load(file_path)
    return data


def read_text_file(file_path: pathlib.Path | str) -> str:
    """
    Reads a text file and returns its entire contents as a single string.
    Includes error checking and logging.

    Args:
    file_path (pathlib.Path | str): The file path of the text file to read.

    Returns:
    str: The entire contents of the text file as a single string.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Successfully read {json.dumps(str(file_path))}")
        return text
    except FileNotFoundError:
        logger.error(f"File not found: {json.dumps(str(file_path))}")
        return ""
    except PermissionError:
        logger.error(f"Permission denied: {json.dumps(str(file_path))}")
        return ""
    except OSError as e:
        logger.error(f"Error reading {json.dumps(str(file_path))}", e)
        return ""


def read_text_file_lines(file_path: pathlib.Path | str) -> list[str]:
    """
    Reads a text file and returns a list of strings with the \n characters at the end of each line removed.
    Includes error checking and logging.

    Args:
    file_path (pathlib.Path | str): The file path of the text file to read.

    Returns:
    list[str]: A list of strings with the \n characters at the end of each line removed.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "r", encoding="utf-8") as f:
            text_lines = [line.rstrip("\n\r") for line in f]
        logger.info(f"Successfully read {json.dumps(str(file_path))}")
        return text_lines
    except FileNotFoundError:
        logger.error(f"File not found: {json.dumps(str(file_path))}")
        return []
    except PermissionError:
        logger.error(f"Permission denied: {json.dumps(str(file_path))}")
        return []
    except OSError as e:
        logger.error(f"Error reading {json.dumps(str(file_path))}", e)
        return []


def write_text_file(file_path: pathlib.Path | str, text: str) -> None:
    """
    Writes a string to a text file. Includes error checking and logging.

    Args:
    file_path (pathlib.Path | str): The file path of the text file to write.
    text (str): A string to write to the text file.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Successfully wrote {json.dumps(str(file_path))}")
    except FileNotFoundError:
        logger.error(f"File not found: {json.dumps(str(file_path))}")
    except PermissionError:
        logger.error(f"Permission denied: {json.dumps(str(file_path))}")
    except OSError as e:
        logger.error(f"Error writing {json.dumps(str(file_path))}", e)


def write_text_file_lines(file_path: pathlib.Path | str, lines: list[str]) -> None:
    """
    Writes a list of strings to a text file, with each string on a new line.
    Includes error checking and logging.

    Args:
    file_path (pathlib.Path | str): The file path of the text file to write.
    lines (list[str]): A list of strings to write to the text file.

    Returns:
    None
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logger.info(f"Successfully wrote {json.dumps(str(file_path))}")
    except FileNotFoundError:
        logger.error(f"File not found: {json.dumps(str(file_path))}")
    except PermissionError:
        logger.error(f"Permission denied: {json.dumps(str(file_path))}")
    except OSError as e:
        logger.error(f"Error writing {json.dumps(str(file_path))}", e)


def read_json_file(file_path: pathlib.Path | str) -> dict | list:
    """
    Reads a json file as a dictionary or list. Includes error checking and logging.

    Args:
    file_path (pathlib.Path | str): The file path of the json file to read.

    Returns:
    dict | list: The contents of the json file as a dictionary or list.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            logger.debug(f"Successfully read json file: {json.dumps(str(file_path))}")
            return json_data
    except FileNotFoundError:
        logger.error(f"File not found: {json.dumps(str(file_path))}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding json file: {json.dumps(str(file_path))}")
        raise


def write_json_file(data: dict | list, file_path: pathlib.Path | str) -> None:
    """
    Writes a dictionary or list to a json file. Includes error checking and logging.

    Args:
    data (dict | list): The data to write to the json file.
    file_path (pathlib.Path | str): The file path of the json file to write.

    Returns:
    None
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {json.dumps(str(file_path.parent))}")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
            logger.info(f"Successfully wrote json file: {json.dumps(str(file_path))}")
    except IOError:
        logger.error(f"Error writing json file: {json.dumps(str(file_path))}")
        raise


class FileMetadata(NamedTuple):
    """Container for file statistics."""
    modified: int
    created: int
    size: int


def get_file_data(file_path: pathlib.Path | str) -> FileMetadata:
    """
    Gets file stats safely across different Operating Systems.
    Returns times in nanoseconds.
    """
    path = pathlib.Path(file_path)
    stat = path.stat()
    mtime = stat.st_mtime_ns
    try:
        # macOS/BSD
        ctime = getattr(stat, 'st_birthtime_ns', None)
        if ctime is None:
            # Windows (st_ctime is creation time on Windows)
            ctime = stat.st_ctime_ns
    except AttributeError:
        # Linux fallback (st_ctime is metadata change, not birth)
        ctime = mtime

    size = stat.st_size
    logger.debug(f"File stats for {path.name}: {mtime=}, {ctime=}, {size=}")
    return FileMetadata(modified=mtime, created=ctime, size=size)


def is_in_tolerance(
    experimental_value: float,
    target_value: float,
    target_tolerance: float,
    name: str
) -> bool:
    """Check if a measured value falls within the target range."""
    # Logic
    deviation = round(abs(experimental_value - target_value), 2)
    in_tolerance = (target_value - target_tolerance) <= experimental_value <= (target_value + target_tolerance)

    # Logging setup
    symbol = ">" if deviation > target_tolerance else "<" if deviation < target_tolerance else "="
    status = "PASS" if in_tolerance else "FAIL"

    log_msg = f"{name} | {status} | Measured: {experimental_value} | Target: {target_value}±{target_tolerance} | Dev: {deviation} {symbol} {target_tolerance}"

    if in_tolerance:
        logger.info(log_msg)
    else:
        logger.warning(log_msg)

    return in_tolerance


class Styles:
    """Color and formatting styles"""
    # Text Effects
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\x1B[3m"
    UNDERLINED = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

    # Standard Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright Colors (Your original list mostly used these)
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_PURPLE = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Backgrounds
    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    PURPLE_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"

    @classmethod
    def preview_styles(cls):
        """Preview all available styles"""
        for each in [style for style in vars(cls) if not style.startswith("__")]:
            print(f"{each}: {getattr(cls, each)}ABCabc#@!?0123{cls.RESET}")


class ColorFormatter(logging.Formatter):
    """Adds colors to specific keywords for console output only."""

    def format(self, record):
        message = super().format(record)
        if "PASS" in message:
            message = message.replace("PASS", f"{Styles.GREEN_BG}{Styles.BOLD}PASS{Styles.RESET}")
        elif "FAIL" in message:
            message = message.replace("FAIL", f"{Styles.RED_BG}{Styles.BOLD}FAIL{Styles.RESET}")

        return message


def main() -> None:
    """
    This is where your main script logic goes
    """


def format_duration_long(duration_seconds: float) -> str:
    """
    Format duration in a human-friendly way, showing only the two largest non-zero units.
    For durations >= 1s, do not show microseconds or nanoseconds.
    For durations >= 1m, do not show milliseconds.
    """
    ns = int(duration_seconds * 1_000_000_000)
    units = [
        ("y", 365 * 24 * 60 * 60 * 1_000_000_000),
        ("mo", 30 * 24 * 60 * 60 * 1_000_000_000),
        ("d", 24 * 60 * 60 * 1_000_000_000),
        ("h", 60 * 60 * 1_000_000_000),
        ("m", 60 * 1_000_000_000),
        ("s", 1_000_000_000),
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ]
    parts = []
    for name, factor in units:
        value, ns = divmod(ns, factor)
        if value:
            parts.append(f"{value}{name}")
        if len(parts) == 2:
            break
    if not parts:
        return "0s"
    return "".join(parts)


def enforce_max_log_count(dir_path: pathlib.Path | str, max_count: int | None, script_name: str) -> None:
    """
    Keep only the N most recent logs for this script.
    """
    if max_count is None or max_count <= 0:
        return

    dir_path = pathlib.Path(dir_path)

    # Get all logs for this script, sorted by name (which is our timestamp)
    # Newest will be at the end of the list
    files = sorted([f for f in dir_path.glob(f"*{script_name}*.log") if f.is_file()])

    # If we have more than the limit, calculate how many to delete
    if len(files) > max_count:
        to_delete = files[:-max_count]  # Everything except the last N files
        for f in to_delete:
            try:
                f.unlink()
                logger.debug(f"Deleted old log: {f.name}")
            except OSError as e:
                logger.error(f"Failed to delete {f.name}: {e}")


def setup_logging(
        logger_obj: logging.Logger,
        file_path: pathlib.Path | str,
        script_name: str,
        max_log_files: int | None = None,
        console_logging_level: int = logging.DEBUG,
        file_logging_level: int = logging.DEBUG,
        message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Set up logging for a script.

    Args:
    logger_obj (logging.Logger): The logger object to configure.
    file_path (pathlib.Path | str): The file path of the log file to write.
    max_log_files (int | None, optional): The maximum total size for all logs in the folder. Defaults to None.
    console_logging_level (int, optional): The logging level for console output. Defaults to logging.DEBUG.
    file_logging_level (int, optional): The logging level for file output. Defaults to logging.DEBUG.
    message_format (str, optional): The format string for log messages. Defaults to "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s".
    date_format (str, optional): The format string for log timestamps. Defaults to "%Y-%m-%d %H:%M:%S".
    """

    file_path = pathlib.Path(file_path)
    dir_path = file_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    logger_obj.handlers.clear()
    logger_obj.setLevel(file_logging_level)

    formatter = logging.Formatter(message_format, datefmt=date_format)

    # File Handler
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(file_logging_level)
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(ColorFormatter(message_format, datefmt=date_format))
    logger_obj.addHandler(console_handler)

    if max_log_files is not None:
        enforce_max_log_count(dir_path, max_log_files, script_name)


def load_config(file_path: pathlib.Path | str) -> dict:
    """
    Load configuration from a TOML file.

    Args:
    file_path (pathlib.Path | str): The file path of the TOML file to read.

    Returns:
    dict: The contents of the TOML file as a dictionary.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {json.dumps(str(file_path))}")
    data = read_toml(file_path)
    return data


def bootstrap():
    """
    Handles environment setup, configuration loading,
    and logging before executing the main script logic.
    """
    exit_code = 0
    try:
        # Resolve paths and configuration
        script_path = pathlib.Path(__file__)
        script_name = script_path.stem
        config_path = script_path.with_name(f"{script_name}_config.toml")

        # Load settings
        config = load_config(config_path)
        logger_config = config.get("logging", {})

        # Parse log levels and formats
        console_log_level = getattr(logging, logger_config.get("console_logging_level", "INFO").upper(), logging.INFO)
        file_log_level = getattr(logging, logger_config.get("file_logging_level", "INFO").upper(), logging.INFO)
        log_message_format = logger_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s")

        # Setup directories and filenames
        logs_folder = pathlib.Path(logger_config.get("logs_folder_name", "logs"))
        logs_folder.mkdir(parents=True, exist_ok=True)

        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = logs_folder / f"{timestamp}__{script_name}__{pc_name}.log"

        # Initialize logging
        setup_logging(
            logger_obj=logger,
            file_path=log_path,
            script_name=script_name,
            max_log_files=logger_config.get("max_log_files"),
            console_logging_level=console_log_level,
            file_logging_level=file_log_level,
            message_format=log_message_format
        )

        start_ns = time.perf_counter_ns()
        logger.info(f"Script: {json.dumps(script_name)} | Version: {__version__} | Host: {json.dumps(pc_name)}")

        main()

        end_ns = time.perf_counter_ns()
        duration_str = format_duration_long((end_ns - start_ns) / 1e9)
        logger.info(f"Execution completed in {duration_str}.")

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Using 'err' or 'exc' is standard; logging the traceback handles the 'broad-except'
        logger.error("A fatal error has occurred: %r", e, exc_info=True)
        exit_code = 1
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
