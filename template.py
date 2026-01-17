"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import ctypes
import json
import logging
import math
# import os
import pathlib
import socket
import sys
import time
# import traceback
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple, Union, Optional

import numpy as np
import sympy as sp
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

    logger.info(log_msg)
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
        """Preview all available styles, skipping methods and internal attributes."""
        # Get all attributes that are strings and don't start with underscore
        style_names = [
            name for name in dir(cls)
            if not name.startswith("_") and isinstance(getattr(cls, name), str)
        ]

        for name in sorted(style_names):
            logger.debug(f"{name.ljust(15)}: {getattr(cls, name)}ABCabc#@!?0123{cls.RESET}")


class Unit:
    """Lightweight unit descriptor with conversion support"""

    def __init__(self, name: str, dimension: str, factor_to_base: float = 1.0):
        self.name = name
        self.dimension = dimension
        self.factor_to_base = factor_to_base  # scale to base unit (g, mm, s)

    def __str__(self):
        return self.name if self.name else "1"

    def __mul__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        if self.is_dimensionless():
            return other
        if other.is_dimensionless():
            return self
        return Unit(
            f"{self.name}·{other.name}" if self.name and other.name else self.name or other.name,
            f"{self.dimension}·{other.dimension}" if self.dimension and other.dimension else self.dimension or other.dimension,
            self.factor_to_base * other.factor_to_base
        )

    def __truediv__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        if self.factor_to_base == other.factor_to_base and self.dimension == other.dimension:
            return Unit.dimensionless()
        if other.is_dimensionless():
            return self
        if self.is_dimensionless():
            return Unit(f"1/{other.name}", f"1/{other.dimension}", 1 / other.factor_to_base)
        return Unit(
            f"{self.name}/{other.name}" if self.name and other.name else self.name or other.name,
            f"{self.dimension}/{other.dimension}" if self.dimension and other.dimension else self.dimension or other.dimension,
            self.factor_to_base / other.factor_to_base
        )

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return False
        return self.name == other.name and self.dimension == other.dimension

    def is_dimensionless(self):
        return self.name == "" or self.dimension == "dimensionless"

    # ---------------- Unit constructors ----------------
    @classmethod
    def dimensionless(cls):
        return cls("", "dimensionless", 1.0)

    # Mass
    @classmethod
    def kilogram(cls):
        return cls("kg", "mass", 1000.0)

    @classmethod
    def gram(cls):
        return cls("g", "mass", 1.0)

    @classmethod
    def milligram(cls):
        return cls("mg", "mass", 0.001)

    # Length
    @classmethod
    def meter(cls):
        return cls("m", "length", 1000.0)

    @classmethod
    def millimeter(cls):
        return cls("mm", "length", 1.0)

    @classmethod
    def kilometer(cls):
        return cls("km", "length", 1_000_000.0)

    @classmethod
    def centimeter(cls):
        return cls("cm", "length", 10.0)

    @classmethod
    def inch(cls):
        return cls("in", "length", 25.4)

    @classmethod
    def foot(cls):
        return cls("ft", "length", 304.8)

    @classmethod
    def mile(cls):
        return cls("mi", "length", 1_609_344.0)

    # Time
    @classmethod
    def second(cls):
        return cls("s", "time", 1.0)

    @classmethod
    def millisecond(cls):
        return cls("ms", "time", 0.001)

    # Temperature
    @classmethod
    def celsius(cls):
        return cls("°C", "temperature", 1.0)

    @classmethod
    def fahrenheit(cls):
        return cls("°F", "temperature", 1.0)

    @classmethod
    def kelvin(cls):
        return cls("K", "temperature", 1.0)

    # ---------------- Conversion ----------------
    def convert_value_to(self, value: float, target_unit: "Unit", scale_only=False) -> float:
        """Convert a numeric value from this unit to another unit."""
        if self.dimension != target_unit.dimension:
            raise ValueError(f"Cannot convert {self.dimension} to {target_unit.dimension}")

        # Temperature special case
        if self.dimension == "temperature":
            # Convert self -> Kelvin
            match self.name:
                case "°C": val_in_base = value + 273.15
                case "°F": val_in_base = (value - 32) * 5 / 9 + 273.15
                case "K": val_in_base = value
                case _: val_in_base = value

            # Convert Kelvin -> target
            match target_unit.name:
                case "°C": result = val_in_base - 273.15
                case "°F": result = (val_in_base - 273.15) * 9 / 5 + 32
                case "K": result = val_in_base
                case _: result = val_in_base

            if scale_only:
                # Only scale uncertainty linearly
                if self.name == "°C" and target_unit.name == "°F":
                    return value * 9 / 5
                if self.name == "°F" and target_unit.name == "°C":
                    return value * 5 / 9
                return value
            return result

        # Linear conversion
        return value * (self.factor_to_base / target_unit.factor_to_base)


# ---------------- Measurement ----------------
class Measurement:
    """Number with uncertainty, units, and decimals."""

    def __init__(self, value: float, decimals=None, uncertainty: float = 0.0, units: Unit = Unit.dimensionless()):
        if units is None:
            units = Unit.dimensionless()
        self.value = value
        self.uncertainty = uncertainty
        self.units = units
        self.decimals = decimals
        self._round()

    def _round(self):
        if self.decimals is not None:
            self.value = round(self.value, self.decimals)
            self.uncertainty = round(self.uncertainty, self.decimals)

    # ---------------- Arithmetic ----------------
    def _as_measurement(self, other) -> "Measurement":
        if isinstance(other, Measurement):
            return other
        if isinstance(other, (int, float)):
            return Measurement(float(other), decimals=None, uncertainty=0.0, units=self.units)
        raise TypeError(f"Cannot convert {other} to Measurement")

    def _decimals_for(self, other: "Measurement"):
        if self.decimals is None and other.decimals is None:
            return None
        if self.decimals is None:
            return other.decimals
        if other.decimals is None:
            return self.decimals
        return min(self.decimals, other.decimals)

    def __add__(self, other):
        other = self._as_measurement(other)
        if self.units != other.units and not other.units.is_dimensionless():
            raise ValueError(f"Unit mismatch: {self.units} vs {other.units}")
        value = self.value + other.value
        uncertainty = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return Measurement(value, decimals=self._decimals_for(other), uncertainty=uncertainty, units=self.units)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._as_measurement(other)
        if self.units != other.units and not other.units.is_dimensionless():
            raise ValueError(f"Unit mismatch: {self.units} vs {other.units}")
        value = self.value - other.value
        uncertainty = math.sqrt(self.uncertainty**2 + other.uncertainty**2)
        return Measurement(value, decimals=self._decimals_for(other), uncertainty=uncertainty, units=self.units)

    def __rsub__(self, other):
        return self._as_measurement(other) - self

    def __mul__(self, other):
        other = self._as_measurement(other)
        value = self.value * other.value
        rel_unc = math.sqrt(
            (self.uncertainty / self.value if self.value != 0 else 0)**2 +
            (other.uncertainty / other.value if other.value != 0 else 0)**2
        )
        uncertainty = abs(value) * rel_unc
        units = self.units * other.units
        return Measurement(value, decimals=self._decimals_for(other), uncertainty=uncertainty, units=units)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._as_measurement(other)
        value = self.value / other.value
        rel_unc = math.sqrt(
            (self.uncertainty / self.value if self.value != 0 else 0)**2 +
            (other.uncertainty / other.value if other.value != 0 else 0)**2
        )
        uncertainty = abs(value) * rel_unc
        units = self.units / other.units
        return Measurement(value, decimals=self._decimals_for(other), uncertainty=uncertainty, units=units)

    def __rtruediv__(self, other):
        return self._as_measurement(other) / self

    def __pow__(self, exponent):
        value = self.value ** exponent
        rel_unc = abs(exponent) * (self.uncertainty / self.value if self.value != 0 else 0)
        uncertainty = abs(value) * rel_unc
        return Measurement(value, decimals=self.decimals, uncertainty=uncertainty, units=self.units)

    # ---------------- Conversions ----------------
    def convert_to(self, target_unit: Unit) -> "Measurement":
        """Convert value and uncertainty using Unit class."""
        new_value = self.units.convert_value_to(self.value, target_unit)
        new_uncertainty = self.units.convert_value_to(self.uncertainty, target_unit, scale_only=True)
        return Measurement(new_value, self.decimals, new_uncertainty, target_unit)

    # ---------------- Representation ----------------
    def __str__(self):
        dec = self.decimals if self.decimals is not None else max(len(str(self.value).rsplit('.', 1)[-1]), 6)
        val_str = f"{self.value:.{dec}f}"
        unc_str = f" ± {self.uncertainty:.{dec}f}" if self.uncertainty != 0 else ""
        return f"{val_str}{unc_str} {self.units}"


# ---------------- PhysicsTools ----------------
class PhysicsTools:
    @staticmethod
    def average(measurements: list[Measurement]) -> Measurement:
        if not measurements:
            raise ValueError("No measurements provided")

        values = [m.value for m in measurements]
        mean = sum(values) / len(values)
        spread_var = sum((x - mean) ** 2 for x in values) / len(values)
        uncertainty_var = sum(m.uncertainty**2 for m in measurements) / len(measurements)
        total_unc = math.sqrt(spread_var + uncertainty_var)
        decimals = min((m.decimals for m in measurements if m.decimals is not None), default=None)
        units = measurements[0].units
        return Measurement(mean, decimals=decimals, uncertainty=total_unc, units=units)


def main():
    """This is where your main script logic goes"""


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


class ColorFormatter(logging.Formatter):
    """Adds colors to specific keywords for console output only."""

    def format(self, record):
        message = super().format(record)
        if "PASS" in message:
            message = message.replace("PASS", f"{Styles.GREEN_BG}{Styles.BOLD}PASS{Styles.RESET}")
        elif "FAIL" in message:
            message = message.replace("FAIL", f"{Styles.RED_BG}{Styles.BOLD}FAIL{Styles.RESET}")

        return message


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

    # input("Press Enter to exit...")
    return exit_code


if __name__ == "__main__":
    sys.exit(bootstrap())
