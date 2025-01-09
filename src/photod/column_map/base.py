"""
Base class of any column map.
"""

from dataclasses import dataclass


@dataclass
class ColumnMap:

    name: str = "Name of column map"
    purpose: str = "Purpose of map"

    def __init__(self, name: str, purpose: str):
        self.name = name
        self.purpose = purpose
