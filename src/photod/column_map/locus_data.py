"""
Column map for locus data.
"""

from dataclasses import dataclass
from photod.column_map import base

@dataclass
class ColumnMap(base.ColumnMap):

    def __init__(self):
        super().__init__("Catalog column map", "Reference catalog columns")

    Ar: str = "Ar"
    extinction_r: str = Ar
    
    metallicity: str = "FeH"
    FeH: str = "FeH"
    
map = ColumnMap()
