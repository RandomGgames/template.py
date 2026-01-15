import json
import logging
import pathlib
import socket
import sys
import time
import toml
import traceback
import typing
from datetime import datetime

logger = logging.getLogger(__name__)

"""
Python Script Template

Template includes:
- Configurable logging via config file
- Script run time at the end of execution
- Error handling and cleanup
- Total folder size log retention
"""

__version__ = "0.0.0"  # Major.Minor.Patch


def read_toml(file_path: typing.Union[str, pathlib.Path]) -> dict:
    """
    Read configuration settings from the TOML file.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("File not found: {}".format(json.dumps(str(file_path))))
    config = toml.load(file_path)
    return config


def read_text_file(file_path: typing.Union[str, pathlib.Path]) -> str:
    """
    Reads a text file and returns its entire contents as a single string.
    Includes error checking and logging.

    Args:
    file_path (typing.Union[str, pathlib.Path]): The file path of the text file to read.

    Returns:
    str: The entire contents of the text file as a single string.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "r") as f:
            text = f.read()
        logger.info("Successfully read %s", json.dumps(str(file_path)))
        return text
    except Exception as e:
        logger.error("Error reading %s: %s", json.dumps(str(file_path)), e)
        return ""


def read_text_file_lines(file_path: typing.Union[str, pathlib.Path]) -> typing.List[str]:
    """
    Reads a text file and returns a list of strings with the \n characters at the end of each line removed.
    Includes error checking and logging.

    Args:
    file_path (typing.Union[str, pathlib.Path]): The file path of the text file to read.

    Returns:
    typing.List[str]: A list of strings with the \n characters at the end of each line removed.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "r") as f:
            lines = [line.rstrip("\n\r") for line in f]
        logger.info("Successfully read %s", json.dumps(str(file_path)))
        return lines
    except Exception as e:
        logger.error("Error reading %s: %s", json.dumps(str(file_path)), e)
        return []


def write_text_file(file_path: typing.Union[str, pathlib.Path], text: str) -> None:
    """
    Writes a string to a text file. Includes error checking and logging.

    Args:
    file_path (typing.Union[str, pathlib.Path]): The file path of the text file to write.
    text (str): A string to write to the text file.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "w") as f:
            f.write(text)
        logger.info("Successfully wrote %s", json.dumps(str(file_path)))
    except Exception as e:
        logger.error("Error writing %s: %s", json.dumps(str(file_path)), e)


def write_text_file_lines(file_path: typing.Union[str, pathlib.Path], lines: typing.List[str]) -> None:
    """
    Writes a list of strings to a text file, with each string on a new line.
    Includes error checking and logging.

    Args:
    file_path (typing.Union[str, pathlib.Path]): The file path of the text file to write.
    lines (typing.List[str]): A list of strings to write to the text file.

    Returns:
    None
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "r") as f:
            for line in lines:
                f.write(line + "\n")
        logger.info("Successfully wrote %s", json.dumps(str(file_path)))
    except Exception as e:
        logger.error("Error writing %s: %s", json.dumps(str(file_path)), e)


def read_json_file(file_path: typing.Union[str, pathlib.Path]) -> typing.Union[dict, list]:
    """
    Reads a json file as a dictionary or list. Includes error checking and logging.

    Args:
    file_path (typing.Union[str, pathlib.Path]): The file path of the json file to read.

    Returns:
    typing.Union[dict, list]: The contents of the json file as a dictionary or list.
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "r") as file:
            data = json.load(file)
            logger.debug("Successfully read json file: %s", json.dumps(str(file_path)))
            return data
    except FileNotFoundError:
        logger.error("File not found: %s", json.dumps(str(file_path)))
        raise
    except json.JSONDecodeError:
        logger.error("Error decoding json file: %s", json.dumps(str(file_path)))
        raise


def write_json_file(data: typing.Union[dict, list], file_path: typing.Union[str, pathlib.Path]) -> None:
    """
    Writes a dictionary or list to a json file. Includes error checking and logging.

    Args:
    data (typing.Union[dict, list]): The data to write to the json file.
    file_path (typing.Union[str, pathlib.Path]): The file path of the json file to write.

    Returns:
    None
    """
    try:
        file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder: %s", json.dumps(str(file_path.parent)))
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
            logger.info("Successfully wrote json file: %s", json.dumps(str(file_path)))
    except IOError:
        logger.error("Error writing json file: %s", json.dumps(str(file_path)))
        raise


def main() -> None:
    # example_key = config.get("example_key", None)
    pass


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


def enforce_max_folder_size(log_dir: pathlib.Path, max_bytes: int) -> None:
    """
    Enforce a maximum total size for all logs in the folder.
    Deletes oldest logs until below limit.
    """
    if max_bytes is None:
        return

    files = sorted(
        [f for f in log_dir.glob("*.log*") if f.is_file()],
        key=lambda f: f.stat().st_mtime
    )

    total_size = sum(f.stat().st_size for f in files)

    while total_size > max_bytes and files:
        oldest = files.pop(0)
        try:
            size = oldest.stat().st_size
            oldest.unlink()
            logger.debug("Deleted %s", json.dumps(str(oldest)))
            total_size -= size
        except Exception:
            logger.error("Failed to delete %s", json.dumps(str(oldest)), exc_info=True)
            continue


def setup_logging(
        logger: logging.Logger,
        log_file_path: typing.Union[str, pathlib.Path],
        max_folder_size_bytes: typing.Union[int, None] = None,
        console_logging_level: int = logging.DEBUG,
        file_logging_level: int = logging.DEBUG,
        log_message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:

    log_file_path = pathlib.Path(log_file_path)
    log_dir = log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.handlers.clear()
    logger.setLevel(file_logging_level)

    formatter = logging.Formatter(log_message_format, datefmt=date_format)

    # File Handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(file_logging_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if max_folder_size_bytes is not None:
        enforce_max_folder_size(log_dir, max_folder_size_bytes)


def load_config(file_path: typing.Union[str, pathlib.Path]) -> dict:
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("File not found: {}".format(json.dumps(str(file_path))))
    config = read_toml(file_path)
    return config


if __name__ == "__main__":
    error = 0
    try:
        script_name = pathlib.Path(__file__).stem
        config_path = pathlib.Path(f"{script_name}_config.toml")
        # config_path = pathlib.Path("config.toml")
        config = load_config(config_path)

        logging_config = config.get("logging", {})
        console_logging_level = getattr(logging, logging_config.get("console_logging_level", "INFO").upper(), logging.DEBUG)
        file_logging_level = getattr(logging, logging_config.get("file_logging_level", "INFO").upper(), logging.DEBUG)
        log_message_format = logging_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s")
        logs_folder_name = logging_config.get("logs_folder_name", "logs")
        max_folder_size_bytes = logging_config.get("max_folder_size", None)

        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = pathlib.Path(logs_folder_name) / script_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_name = f"{timestamp}_{script_name}_{pc_name}.log"
        log_file_path = log_dir / log_file_name

        setup_logging(
            logger,
            log_file_path,
            max_folder_size_bytes=max_folder_size_bytes,
            console_logging_level=console_logging_level,
            file_logging_level=file_logging_level,
            log_message_format=log_message_format
        )
        start_time = time.perf_counter_ns()
        logger.info("Script: %s | Version: %s | Host: %s", json.dumps(script_name), __version__, json.dumps(pc_name))
        main()
        end_time = time.perf_counter_ns()
        duration = end_time - start_time
        duration = format_duration_long(duration / 1e9)
        logger.info("Execution completed in %s.", duration)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        error = 130
    except Exception as e:
        logger.warning("A fatal error has occurred: %r\n%s", e, traceback.format_exc())
        error = 1
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
        # input("Press Enter to exit...")
        sys.exit(error)
