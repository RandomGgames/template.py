"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import json
import logging
import platform
import re
import send2trash
import socket
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
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
    logs_folder_name: Path = Path(r"Logs")
    console_logging_level: str = "DEBUG"
    file_logging_level: str = "DEBUG"
    log_message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s"
    max_files: int | None = 10

    def __post_init__(self):
        folder = Path(self.logs_folder_name)
        if not folder.is_absolute():
            script_dir = Path(__file__).resolve().parent
            folder = script_dir / folder
        self.logs_folder_name = folder

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
            log_file_name = f"latest_{script_name}__{pc_name}.log"
        elif self.mode == "per_day":
            day_stamp = datetime.now().strftime("%Y%m%d")
            log_file_name = f"{day_stamp}__{script_name}__{pc_name}.log"
        elif self.mode == "per_run":
            log_file_name = f"{timestamp}__{script_name}__{pc_name}.log"
        else:  # Unknown
            log_file_name = f"{timestamp}__{script_name}__{pc_name}.log"

        return log_folder / log_file_name


@dataclass
class ExitBehaviorConfig:
    pause_on_error: bool = True
    always_pause: bool = False


@dataclass
class Config:
    generate_config_if_missing: bool = False
    script: ScriptConfig = field(default_factory=ScriptConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    exit_behavior: ExitBehaviorConfig = field(default_factory=ExitBehaviorConfig)


def main(Config: Config):
    logger.debug("Code goes here")


def save_default_config(file_path: Path, config: Config):
    """Save JSON config safely (atomic write)."""
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    data = convert(asdict(config))
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        temp_path.replace(file_path)  # atomic replace
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def load_config(file_path: Path, generate_if_missing: bool = True) -> Config:
    def merge_defaults(cls, data: dict):
        merged = {**asdict(cls()), **data}
        if "logs_folder_name" in merged:
            merged["logs_folder_name"] = Path(merged["logs_folder_name"])
        return cls(**merged)

    def save_if_needed(config: Config):
        if generate_if_missing:
            save_default_config(file_path, config)

    if not file_path.exists():
        config = Config()
        save_if_needed(config)
        return config

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = file_path.with_name(f"{file_path.stem}.corrupt_{timestamp}.json")
        try:
            file_path.rename(backup)
        except Exception:
            pass

        config = Config()
        save_if_needed(config)
        return config

    except Exception as e:
        print(f"[WARN] Failed to read config: {e}")
        return Config()

    config = Config(
        logging=merge_defaults(LoggingConfig, data.get("logging", {})),
        exit_behavior=merge_defaults(ExitBehaviorConfig, data.get("exit_behavior", {})),
        script=ScriptConfig(),
        generate_config_if_missing=data.get("generate_config_if_missing", True),
    )

    save_if_needed(config)
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
    log_path = config.log_file_path

    logger_obj.handlers.clear()
    logger_obj.setLevel(getattr(logging, config.file_logging_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        config.log_message_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if config.mode == "per_day":
        # Automatic midnight rotation
        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=config.max_files if config.max_files else 0,
            encoding="utf-8",
        )
    else:
        # per_run or single_file
        file_mode = "w" if config.mode == "per_run" else "a"
        file_handler = logging.FileHandler(
            log_path,
            mode=file_mode,
            encoding="utf-8",
        )

    file_handler.setLevel(getattr(logging, config.file_logging_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.console_logging_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)

    try:
        if config.mode in ("per_run", "single_file"):
            if hasattr(config, "SN") and getattr(config, "SN", None):
                script_name_token = f"{Path(__file__).stem}_{config.SN}"
            else:
                script_name_token = Path(__file__).stem

            enforce_max_log_count(
                dir_path=log_path.parent,
                max_count=config.max_files,
                script_name=script_name_token,
            )
    except Exception as pr_err:
        # Do not break logging if pruning fails
        logger_obj.debug("Log pruning skipped due to error: %s", pr_err)

    return log_path


def bootstrap():
    exit_code = 0
    config = Config()
    try:
        script_path = Path(__file__)
        config_path = script_path.with_name(f"{script_path.stem}_config.json")
        config = load_config(config_path, generate_if_missing=Config.generate_config_if_missing)
        tz = datetime.now().astimezone().tzinfo

        setup_logging(logger, config.logging)  # just pass the logging config

        # Pre-main system/environment logging
        logger.debug("Script: %s", json.dumps(script_path.name))
        logger.debug("Version: %s", __version__)
        if hasattr(config.logging, "SN"):
            logger.debug("SN: %s", json.dumps(config.logging.SN))
        logger.debug("Host: %s", json.dumps(socket.gethostname()))
        logger.debug("Current Working Directory: %s", json.dumps(Path.cwd().as_posix()))
        logger.debug("Platform: %s", json.dumps(platform.system()))
        logger.debug("Platform Release: %s", platform.release())
        logger.debug("Platform Version: %s", platform.version())
        logger.debug("Architecture: %s", json.dumps(platform.machine()))
        logger.debug("Python Version: %s", sys.version.split()[0])
        logger.debug("Python Executable: %s", json.dumps(Path(sys.executable).as_posix()))
        logger.debug("Timezone: %s", tz)
        logger.debug("AppConfig: %s", config)
        for name, module in sys.modules.items():
            if module is None or name == "__main__":
                continue
            version = getattr(module, "__version__", None)
            if version is not None:
                logger.debug("Module: %s v%s", name, version)

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
