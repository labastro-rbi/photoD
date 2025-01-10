"""
Column map for catalog data.
"""

from dataclasses import dataclass
from photod.column_map import base


@dataclass
class ColumnMap(base.ColumnMap):

    def __init__(self):
        super().__init__("Catalog column map", "Reference catalog columns")

    # TODO: would like to see these generated from a config, or vice versa
    # TODO: May want these aliases, at least to start with, where each field is its own name too
    # TODO: The RHS are the names in the catalog, not in code

    # Source: https://docs.google.com/document/d/1lDzfDBg_4vd-ces1BhCojwKOlx0oTEmODPClzQbT-zQ/edit#heading=h.olnm3g4r6wj6

    # Est suffix = estimate
    # Unc suffix = uncertainty
    # thus EstUnc = estimated uncertainty as a +/- delta for the quantity it modifies (confidence interval)
    # dS suffix: drop in entropy

    Ar: str = "Ar"
    extinction_r: str = Ar

    ArEst: str = "ArEst"
    ArEstUnc: str = "ArEstUnc"
    ArdS: str = "ArdS"

    metallicity: str = "FeH"
    FeH: str = "FeH"

    FeHEst: str = "FeHEst"
    FeHEstUnc: str = "FeHEstUnc"
    FeHdS: str = "FeHdS"

    Mr: str = "Mr"
    MrEst: str = "MrEst"
    MrEstUnc: str = "MrEstUnc"
    MrdS: str = "MrdS"

    # Qr: absolute magnitude plus extinction
    QrEst: str = "QrEst"
    QrEstUnc: str = "QrEstUnc"

    # I don't see these in the glossary
    chi2min: str = "chi2min"
    glat: str = "glat"
    glon: str = "glon"
    gr: str = "gr"
    rmag: str = "rmag"


map = ColumnMap()
