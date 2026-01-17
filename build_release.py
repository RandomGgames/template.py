"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""


import ast
import ctypes
import importlib.util
import json
import logging
import os
import pathlib
import socket
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import toml

logger = logging.getLogger(__name__)

__version__ = "0.0.0"  # Major.Minor.Patch


def get_common_folder(file_paths):
    return os.path.commonpath(file_paths)


def zip_files(file_paths, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            arcname = os.path.relpath(file, os.path.commonpath(file_paths))
            zipf.write(file, arcname)
    print(f"Created release: {zip_name}")


def find_third_party_imports(file_path: str | Path) -> set[str]:
    file_path = Path(file_path)
    tree = ast.parse(file_path.read_text())

    stdlib = sys.stdlib_module_names
    third_party = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split('.')[0]
                if name not in stdlib:
                    third_party.add(name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split('.')[0]
                if name not in stdlib:
                    third_party.add(name)

    return third_party


def write_requirements(non_std_modules):
    if non_std_modules:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            for module in sorted(non_std_modules):
                f.write(f"{module}\n")
        print("Generated requirements.txt")
    else:
        print("No non-standard modules detected; requirements.txt not generated.")


def main(config):
    validate_config(config, ["files", "name", "version"])

    files = config.get("files")
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise TypeError(f"Expected list of strings for key 'files', got {type(files).__name__}")
    name = str(config.get("name"))
    version = str(config.get("version"))
    if version.startswith("v"):
        version = version[1:]

    # Ensure all files exist
    for file in files:
        if not os.path.isfile(file):
            print(f"Error: File not found: {file}")
            sys.exit(1)

    # Generate zip
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{name}_v{version}_{timestamp}.zip"

    zip_files(files, zip_name)

    # Detect non-standard modules from Python files
    py_files = [f for f in files if f.endswith(".py")]
    non_std_modules = set()
    for f in py_files:
        non_std_module = find_third_party_imports(f)
        non_std_modules.update(non_std_module)
    write_requirements(non_std_modules)


def validate_config(config: dict, required_keys: list[str], nested: bool = False) -> None:
    """
    Validate that all required keys exist in the config dictionary.

    Args:
        config (dict): Configuration dictionary to validate.
        required_keys (Iterable[str]): Keys that must exist in the config.
            For nested keys, use dot notation like "database.host".
        nested (bool): If True, interpret keys with dot notation as nested dictionaries.

    Raises:
        KeyError: If a required key is missing.
    """
    missing_keys = []

    for key in required_keys:
        if nested and '.' in key:
            parts = key.split('.')
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing_keys.append(key)
                    break
        else:
            if key not in config:
                missing_keys.append(key)

    if missing_keys:
        raise KeyError(f"Missing required config keys: {', '.join(missing_keys)}")


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
    """Keep only the N most recent logs for this script."""
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
    console_handler.setFormatter(formatter)
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


def enable_windows_ansi():
    """Enables ANSI escape sequences in the Windows command prompt."""
    kernel32 = ctypes.windll.kernel32
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    # 7 is the standard handle for STDOUT
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


def bootstrap():
    """
    Handles environment setup, configuration loading,
    and logging before executing the main script logic.
    """
    exit_code = 0
    try:
        enable_windows_ansi()

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

        main(config)

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

    # input("Press Enter to exit...")
    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
