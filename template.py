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
import typing
from dataclasses import dataclass, asdict, field
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)

__version__ = "0.0.0"  # Major.Minor.Patch


@dataclass
class ScriptSettings:
    """Place any code for whatever script is being writen here"""


@dataclass
class LogSettings:
    mode: typing.Literal["per_run", "latest", "per_day", "single_file", "ConsoleOnly"] = "per_run"
    folder: Path = Path.home() / "Logs"
    console_level: int = logging.DEBUG
    file_level: int = logging.DEBUG
    date_format: str = "%Y-%m-%d %H:%M:%S"
    message_format: str = "%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] - %(message)s"
    max_files: int | None = 10
    open_log_after_run: bool = False


@dataclass
class RuntimeSettings:
    pause_on_error: bool = True
    always_pause: bool = False


@dataclass
class GlobalSettings:
    use_config_file: bool = False


@dataclass
class Config:
    script: ScriptSettings = field(default_factory=ScriptSettings)
    logs: LogSettings = field(default_factory=LogSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


def main():
    """Code goes here"""


def save_config(file_path: Path, config: Config) -> None:
    """Save config to JSON using atomic write."""

    def to_json_safe(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")

    data = to_json_safe(asdict(config))

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        temp_path.replace(file_path)

    except Exception as e:
        logger.exception("")
        raise e

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def load_config(file_path: Path, generate_if_missing: bool = True) -> Config:
    """Load config, only adding missing defaults. Never overwrite existing values."""

    def merge_missing(default_obj, data: dict):
        """Fill missing keys only. Do not modify existing values."""
        result = {}
        changed = False

        for field_name, default_value in default_obj.__dict__.items():
            if field_name.startswith("_"):
                continue

            if field_name in data:
                incoming = data[field_name]

                # Nested dataclass
                if hasattr(default_value, "__dict__") and isinstance(incoming, dict):
                    merged_child, child_changed = merge_missing(default_value, incoming)
                    result[field_name] = merged_child
                    if child_changed:
                        changed = True
                else:
                    # Convert string → Path if default is Path
                    if isinstance(default_value, Path) and isinstance(incoming, str):
                        incoming = Path(incoming)
                    result[field_name] = incoming
            else:
                # Only place we modify → add missing default
                result[field_name] = default_value
                changed = True

        return type(default_obj)(**result), changed

    def save(cfg: Config):
        if generate_if_missing:
            save_config(file_path, cfg)

    # File missing
    if not file_path.exists():
        config = Config()
        save(config)
        return config

    # Load JSON
    try:
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

    except json.JSONDecodeError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = file_path.with_name(f"{file_path.stem}.corrupt_{timestamp}.json")
        try:
            file_path.rename(backup)
        except Exception:
            pass

        config = Config()
        save(config)
        return config

    except:
        logger.exception("Failed to read config")
        return Config()

    # Merge (ONLY missing keys)
    config = Config()

    logs, logs_changed = merge_missing(config.logs, raw_data.get("logs", {}))
    script, script_changed = merge_missing(config.script, raw_data.get("script", {}))
    runtime, runtime_changed = merge_missing(config.runtime, raw_data.get("runtime", {}))

    config = Config(
        logs=logs,
        script=script,
        runtime=runtime,
    )

    # Detect if anything was added
    any_changed = logs_changed or script_changed or runtime_changed

    # Save only if defaults were added
    if any_changed:
        save(config)

    return config


def enforce_max_log_count(
    dir_path: Path,
    max_count: int,
    script_name: str,
) -> None:
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


logger = logging.getLogger(__name__)
log_buffer = logging.handlers.MemoryHandler(
    capacity=0,
    flushLevel=logging.CRITICAL,
    target=None,
)
logger.addHandler(log_buffer)
logger.setLevel(logging.DEBUG)


def setup_logging(logger_obj: logging.Logger, log_settings: LogSettings) -> Path | None:
    """Set up file and console logging with flexible modes and rotation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    day_stamp = datetime.now().strftime("%Y%m%d")
    script_name = Path(__file__).stem
    pc_name = socket.gethostname()

    log_path: Path | None = None

    if log_settings.mode != "ConsoleOnly":
        log_dir = (log_settings.folder if isinstance(log_settings.folder, Path) else Path(log_settings.folder))
        log_dir = log_dir.expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created log folder: %s", log_dir.as_posix())

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

    # Flush logs buffer from prior to logging initialization
    if "log_buffer" in globals():
        class _ForwardToLogger(logging.Handler):
            def emit(self, record):
                logger_obj.handle(record)

        forward_handler = _ForwardToLogger()
        log_buffer.setTarget(forward_handler)
        log_buffer.flush()
        log_buffer.close()

    # Enforce max log count (except per_day which rotates automatically)
    if log_settings.max_files and log_path and log_settings.mode not in ("per_day", "ConsoleOnly"):
        try:
            enforce_max_log_count(
                dir_path=log_path.parent,
                max_count=log_settings.max_files,
                script_name=script_name,
            )
        except Exception as e:
            logger_obj.debug("Log pruning skipped: %s", e)

    return log_path


def bootstrap():
    exit_code = 0
    config = Config()
    log_path = None
    script_path = Path(__file__)
    config_path = script_path.with_name(f"{script_path.stem}_config.json")
    global_settings = GlobalSettings()
    try:
        config = load_config(config_path, generate_if_missing=global_settings.use_config_file)
        log_path = setup_logging(logger_obj=logger, log_settings=config.logs)
        logger.debug("Script: %s", json.dumps(script_path.name))
        logger.debug("Version: %s", __version__)
        logger.debug("Host: %s", json.dumps(socket.gethostname()))
        logger.debug("Current Working Directory: %s", json.dumps(Path.cwd().as_posix()))
        logger.debug("Platform: %s %s v%s", platform.system(), platform.release(), platform.version())
        logger.debug("Python Version: %s", sys.version.split()[0])
        logger.debug("AppConfig: %s", config)
        # for name, module in sys.modules.items():
        #     if module is None or name == "__main__":
        #         continue
        #     version = getattr(module, "__version__", None)
        #     if version is not None:
        #         logger.debug("Module: %s v%s", name, version)

        main()

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

    if config.logs.open_log_after_run and log_path and log_path.exists():
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

    if config.runtime.always_pause or (config.runtime.pause_on_error and exit_code != 0):
        input("Press Enter to exit...")

    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
