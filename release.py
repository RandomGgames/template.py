"""
Python Project Release Builder

This script creates a versioned release archive for a Python project based on
a TOML configuration file. It bundles specified files into a ZIP archive,
automatically detects third-party Python dependencies, and generates a
requirements.txt file listing those dependencies.

Features:
- ZIP release generation with timestamped versioning
- Automatic detection of non-stdlib Python imports
- Optional generation of requirements.txt
- Configurable logging with file rotation
- Windows-first design with cross-platform compatibility

Usage:
1. Create a configuration file named <script_name>_config.toml alongside this script.
2. Define the files to include, project name, version, and optional logging settings.
3. Run the script directly:
       python <script_name>.py

The script will produce:
- A ZIP archive named: <name>_v<version>_<timestamp>.zip
- A requirements.txt file (if third-party imports are detected)
- Log files in the configured logs directory
"""

import ast
import ctypes
import json
import logging
import os
import re
import socket
import sys
import time
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)

__version__ = "1.0.0"  # Major.Minor.Patch


def zip_files(files: list[Path], zip_path: str | Path, compresslevel: int = 6) -> None:
    try:
        paths = [Path(p) for p in files]
        zip_path = Path(zip_path)

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
        ) as zipf:
            for path in paths:
                zipf.write(path, arcname=path.name)

        logger.info("Created release archive: %s", zip_path)

    except Exception as e:
        logger.exception(f"Failed to create zip archive {json.dumps(str(zip_path))}")
        raise


def find_third_party_imports(file_path: str | Path) -> set[str]:
    """
    Parse a Python source file and return the names of imported third-party
    (non-stdlib) top-level modules.

    Args:
        file_path: Path to a Python source file.

    Returns:
        A set of top-level module names that are not part of the standard library.

    Raises:
        FileNotFoundError: If the file does not exist.
        SyntaxError: If the Python file cannot be parsed.
        OSError: If the file cannot be read.
    """
    try:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        stdlib = sys.stdlib_module_names
        third_party: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.partition(".")[0]
                    if name not in stdlib:
                        third_party.add(name)

            elif isinstance(node, ast.ImportFrom) and node.module:
                name = node.module.partition(".")[0]
                if name not in stdlib:
                    third_party.add(name)

        return third_party

    except Exception as e:
        logger.exception(f"Failed to analyze imports in {json.dumps(str(file_path))}")
        raise


def write_requirements(non_std_modules: set[str], output_path: str | Path = "requirements.txt") -> None:
    """
    Write a requirements.txt file containing the given set of third-party modules.

    Args:
        non_std_modules: Set of third-party module names.
        output_path: Path to the requirements file to generate.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        if not non_std_modules:
            logger.info(f"No non-standard modules detected; {json.dumps(str(output_path))} not generated")
            return

        path = Path(output_path)

        with path.open("w", encoding="utf-8", newline="\n") as f:
            for module in sorted(non_std_modules):
                f.write(f"{module}\n")

        logger.info(f"Generated requirements file: {json.dumps(str(path))}")

    except Exception:
        logger.exception(f"Failed to write requirements file to {json.dumps(str(output_path))}")
        raise


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.

    Removes characters that are illegal on Windows: <>:"/\\|?*
    Trims leading and trailing whitespace.
    Replaces sequences of illegal characters with an underscore.

    Args:
        name: The raw filename string.

    Returns:
        A safe filename string.
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected str, got {type(name).__name__}")

    # Define illegal characters for Windows
    illegal_chars = r'[<>:"/\\|?*]'

    # Replace illegal characters with underscore
    sanitized = re.sub(illegal_chars, "_", name)

    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()

    # Optional: collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Avoid empty string
    if not sanitized:
        sanitized = "file"

    return sanitized


def main(config):
    name = str(config.get("name"))

    version = str(config.get("version"))
    if version.startswith("v"):
        version = version[1:]

    config_files = config.get("files")
    if not isinstance(config_files, list) or not all(isinstance(f, str) for f in config_files):
        raise TypeError("Expected list of strings for key 'files'")
    files = [Path(f) for f in config_files]

    # Ensure all files exist
    for file in files:
        if not Path(file).is_file():
            print(f"Error: File not found: {file}")
            sys.exit(1)

    # Generate zip
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{name}_v{version}_{timestamp}.zip"
    zip_name = zip_name.replace(" ", "_")
    zip_name = sanitize_filename(zip_name)

    zip_files(files, zip_name)

    # Detect non-standard modules from Python files
    py_files = [f for f in files if f.suffix == ".py"]
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


def read_toml(file_path: Path | str) -> dict:
    """
    Reads a TOML file and returns its contents as a dictionary.

    Args:
        file_path (Path | str): The file path of the TOML file to read.

    Returns:
        dict: The contents of the TOML file as a dictionary.

    Raises:
        FileNotFoundError: If the TOML file does not exist.
        OSError: If the file cannot be read.
        tomllib.TOMLDecodeError (or toml.TomlDecodeError): If the file is invalid TOML.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {json.dumps(str(path))}")

    try:
        # Read TOML as bytes
        with path.open("rb") as f:
            data = tomllib.load(f)  # Replace with 'toml.load(f)' if using the toml package
        return data

    except (OSError, tomllib.TOMLDecodeError):
        logger.exception(f"Failed to read TOML file: {json.dumps(str(file_path))}")
        raise


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


def enforce_max_log_count(dir_path: Path | str, max_count: int | None, script_name: str) -> None:
    """Keep only the N most recent logs for this script."""
    if max_count is None or max_count <= 0:
        return

    dir_path = Path(dir_path)

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
        file_path: Path | str,
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
    file_path (Path | str): The file path of the log file to write.
    max_log_files (int | None, optional): The maximum total size for all logs in the folder. Defaults to None.
    console_logging_level (int, optional): The logging level for console output. Defaults to logging.DEBUG.
    file_logging_level (int, optional): The logging level for file output. Defaults to logging.DEBUG.
    message_format (str, optional): The format string for log messages. Defaults to "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s]: %(message)s".
    date_format (str, optional): The format string for log timestamps. Defaults to "%Y-%m-%d %H:%M:%S".
    """

    file_path = Path(file_path)
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


def load_config(file_path: Path | str) -> dict:
    """
    Load configuration from a TOML file.

    Args:
    file_path (Path | str): The file path of the TOML file to read.

    Returns:
    dict: The contents of the TOML file as a dictionary.
    """
    file_path = Path(file_path)
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
        if os.name == "nt":
            enable_windows_ansi()

        # Resolve paths and configuration
        script_path = Path(__file__)
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
        logs_folder = Path(logger_config.get("logs_folder_name", "logs"))
        logs_folder.mkdir(parents=True, exist_ok=True)

        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
