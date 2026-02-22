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
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.0.0"  # Major.Minor.Patch


@dataclass
class ScriptConfig:
    pass


@dataclass
class LoggingConfig:
    mode: str = "per_run"  # single_file | per_run | per_day
    # SN: str = field(default_factory=lambda: input("Please input the SN: "))
    logs_folder_name: str = "logs"
    console_logging_level: str = "DEBUG"
    file_logging_level: str = "DEBUG"
    log_message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s"
    max_files: int | None = 10

    def __post_init__(self):
        if hasattr(self, "SN"):
            sn = (self.SN or "").strip().replace(" ", "_").upper()
            sn = re.sub(r'[<>:"/\\|?*]', "", sn)
            self.SN = sn if sn else "Undefined"

    @property
    def log_file_path(self) -> Path:
        """Generate log path based on mode and optional SN folder."""
        # Base folder
        base_folder = Path(self.logs_folder_name)
        if hasattr(self, "SN") and getattr(self, "SN", None) not in ("", None):
            log_folder = base_folder / self.SN
        else:
            log_folder = base_folder
        log_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pc_name = socket.gethostname()
        if hasattr(self, "SN") and getattr(self, "SN", None) not in ("", None):
            script_name = Path(__file__).stem + "_" + self.SN
        else:
            script_name = Path(__file__).stem

        # Filename depends on mode
        if self.mode == "single_file":
            log_file_name = f"{script_name}__{pc_name}.log"
        elif self.mode == "per_day":
            day_stamp = datetime.now().strftime("%Y%m%d")
            log_file_name = f"{day_stamp}__{script_name}__{pc_name}.log"
        else:  # per_run
            log_file_name = f"{timestamp}__{script_name}__{pc_name}.log"

        return log_folder / log_file_name


@dataclass
class ExitBehaviorConfig:
    pause_on_error: bool = True
    always_pause: bool = False


@dataclass
class AppConfig:
    script: ScriptConfig = field(default_factory=ScriptConfig)
    generate_config_if_missing: bool = False
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    exit_behavior: ExitBehaviorConfig = field(default_factory=ExitBehaviorConfig)


def main(config: AppConfig):
    logger.debug("Code goes here")


def save_default_config(file_path: Path, config: AppConfig):
    """Save JSON config based on dataclass defaults."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)


def load_config(file_path: Path, generate_if_missing: bool = True) -> AppConfig:
    """
    Load configuration from JSON file.
    If missing and generate_if_missing=True, auto-generate.
    If missing and generate_if_missing=False, use dataclass defaults.
    Any missing keys in existing JSON are filled from dataclass defaults.
    """
    def merge_defaults(dataclass_type, data: dict):
        """Fill missing keys from dataclass defaults."""
        defaults = asdict(dataclass_type())
        merged = {**defaults, **data}  # Python 3.9+ can also use defaults | data
        return dataclass_type(**merged)

    if not file_path.exists():
        config = AppConfig()
        if generate_if_missing:
            save_default_config(file_path, config)
        return config

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Merge defaults for each section
    logging_cfg = merge_defaults(LoggingConfig, data.get("logging", {}))
    exit_cfg = merge_defaults(ExitBehaviorConfig, data.get("exit_behavior", {}))

    # ScriptConfig is currently empty; merge if fields are added later
    script_cfg = ScriptConfig()

    config = AppConfig(
        logging=logging_cfg,
        exit_behavior=exit_cfg,
        script=script_cfg,
        generate_config_if_missing=data.get("generate_config_if_missing", True)
    )

    # Optionally overwrite JSON with missing keys filled in
    if generate_if_missing:
        save_default_config(file_path, config)

    return config


def enforce_max_log_count(dir_path: Path | str, max_count: int | None, script_name: str):
    """
    Keep only the N most recent logs for this script.

    Oldest files are deleted first, based on modification time.

    Args:
        dir_path (Path | str): Directory containing log files.
        max_count (int | None): Maximum number of log files to keep. None = no limit.
        script_name (str): Filter logs by this script name.
    """
    if max_count is None or max_count <= 0:
        return

    dir_path = Path(dir_path)
    # Get all matching log files
    files = [f for f in dir_path.glob(f"*{script_name}*.log") if f.is_file()]

    # Sort by modification time (oldest first)
    files.sort(key=lambda f: f.stat().st_mtime)

    # Delete excess files
    if len(files) > max_count:
        to_delete = files[:len(files) - max_count]  # oldest files
        for f in to_delete:
            try:
                send2trash.send2trash(f)
                logger.debug(f"Deleted old log: {f.name}")
            except OSError as e:
                logger.error(f"Failed to delete {f.name}: {e}")


def setup_logging(logger_obj: logging.Logger, config: LoggingConfig) -> Path:
    """Set up file and console logging based on AppConfig."""
    log_path = config.log_file_path  # use the property

    logger_obj.handlers.clear()
    logger_obj.setLevel(getattr(logging, config.file_logging_level.upper(), logging.INFO))
    formatter = logging.Formatter(config.log_message_format, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    file_mode = "a" if config.mode == "per_day" else "w"  # append for per_day
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(getattr(logging, config.file_logging_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.console_logging_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)

    return log_path


def bootstrap():
    exit_code = 0
    try:
        script_path = Path(__file__)
        config_path = script_path.with_name(f"{script_path.stem}_config.json")
        config = load_config(config_path, generate_if_missing=AppConfig.generate_config_if_missing)

        setup_logging(logger, config.logging)  # just pass the logging config

        logger.info(f"Script: {script_path.stem}")
        logger.info(f"Version: {__version__}")
        logger.info(f"Host: {socket.gethostname()}")
        logger.info(f"AppConfig: {config}")

        main(config)

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        exit_code = 130
    except Exception as e:
        logger.exception(f"A fatal error has occurred: {e}")
        exit_code = 1
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if config.exit_behavior.always_pause or (config.exit_behavior.pause_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
