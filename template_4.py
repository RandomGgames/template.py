"""
{Script Name}

{Summary of what the script does}

{How to use the script}
"""

import logging


__version__ = "0.0.0"  # Major.Minor.Patch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S",)
logger = logging.getLogger(__name__)


def main():
    """Code goes here."""

if __name__ == "__main__":
    main()
