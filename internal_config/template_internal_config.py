"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import json
import logging
import send2trash
import socket
import sys
import tomllib
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__version__ = "0.0.0"  # Major.Minor.Patch

CONFIG = {
    "logging": {
        "console_logging_level": "DEBUG",
        "file_logging_level": "DEBUG",
        "max_log_files": 50,
    },
    "exit_behavior": {
        "always_pause": False,
        "pause_on_error": True,
    }
}


def main():
    """This is where your main script logic goes"""


def enforce_max_log_count(dir_path: Path | str, max_count: int | None, script_name: str) -> None:
    """
    Keep only the N most recent logs for this script.

    Args:
        dir_path (Path | str): The directory path to the log files.
        max_count (int | None): The maximum number of log files to keep. None for no limit.
        script_name (str): The name of the script to filter logs by.
    """
    if max_count is None or max_count <= 0:
        return
    dir_path = Path(dir_path)
    files = sorted([f for f in dir_path.glob(f"*{script_name}*.log") if f.is_file()])  # Newest will be at the end of the list
    if len(files) > max_count:
        to_delete = files[:-max_count]  # Everything except the last N files
        for f in to_delete:
            try:
                send2trash.send2trash(f)
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


def bootstrap():
    """
    Handles environment setup, configuration loading,
    and logging before executing the main script logic.
    """
    exit_code = 0
    try:
        script_path = Path(__file__)
        script_name = script_path.stem
        pc_name = socket.gethostname()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger_config = CONFIG.get("logging", {})
        console_log_level = getattr(logging, logger_config.get("console_logging_level", "INFO").upper(), logging.INFO)
        file_log_level = getattr(logging, logger_config.get("file_logging_level", "INFO").upper(), logging.INFO)
        log_message_format = logger_config.get("log_message_format", "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s")
        logs_folder = Path(logger_config.get("logs_folder_name", "logs"))
        log_path = logs_folder / f"{timestamp}__{script_name}__{pc_name}.log"
        setup_logging(
            logger_obj=logger,
            file_path=log_path,
            script_name=script_name,
            max_log_files=logger_config.get("max_log_files"),
            console_logging_level=console_log_level,
            file_logging_level=file_log_level,
            message_format=log_message_format
        )

        exit_behavior_config = CONFIG.get("exit_behavior", {})
        pause_before_exit = exit_behavior_config.get("always_pause", False)
        pause_before_exit_on_error = exit_behavior_config.get("pause_on_error", True)

        logger.info(f"Script: {json.dumps(script_name)} | Version: {__version__} | Host: {json.dumps(pc_name)}")
        main()
        logger.info("Execution completed.")

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Using 'err' or 'exc' is standard; logging the traceback handles the 'broad-except'
        logger.error(f"A fatal error has occurred: {e}")
        exit_code = 1
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if pause_before_exit or (pause_before_exit_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
