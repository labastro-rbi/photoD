"""
Column map for locus data.
"""

from dataclasses import dataclass

from photod.column_map import base


@dataclass
class ColumnMap(base.ColumnMap):
    """Column names of the locus tables."""

    def __init__(self):
        super().__init__("Locus column map", "Reference locus columns")

    abs_mag_r: str = "Mr"
    extinction_r: str = "Ar"
    metallicity: str = "FeH"


map = ColumnMap()
