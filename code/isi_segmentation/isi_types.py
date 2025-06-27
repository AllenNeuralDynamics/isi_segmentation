"""
File that defines the constants used in the package

Renamed from types.py to isi_types.py to avoid conflict with Python standard library 'types' module.
"""

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
