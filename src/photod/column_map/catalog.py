"""
Column map for catalog data.
"""

from dataclasses import dataclass
from photod.column_map import base


@dataclass
class ColumnMap(base.ColumnMap):

    def __init__(self):
        super().__init__("Catalog column map", "Reference catalog columns")

    # Source: https://docs.google.com/document/d/1lDzfDBg_4vd-ces1BhCojwKOlx0oTEmODPClzQbT-zQ/edit#heading=h.olnm3g4r6wj6

    # Est suffix = estimate
    # Unc suffix = uncertainty
    # EstUnc = estimated uncertainty as a +/- delta for the
    #     quantity it modifies (confidence interval)
    # dS suffix: drop in entropy

    # ##### Extinction in r band
    Ar: str = "Ar"
    extinction_r: str = Ar

    ArEst: str = "ArEst"
    extinction_r_est = ArEst

    ArEstUnc: str = "ArEstUnc"
    extinction_r_est_unc = ArEstUnc

    ArdS: str = "ArdS"
    extinction_r_entropy_drop = ArdS

    # ##### Metallicity
    FeH: str = "FeH"
    metallicity: str = FeH

    FeHEst: str = "FeHEst"
    metallicity_est: str = FeHEst

    FeHEstUnc: str = "FeHEstUnc"
    metallicity_est_unc: str = FeHEstUnc

    FeHdS: str = "FeHdS"
    metallicity_entropy_drop: str = FeHdS

    # ##### Absolute magnitude, r band

    Mr: str = "Mr"
    abs_mag_r: str = Mr

    MrEst: str = "MrEst"
    abs_mag_r_est: str = MrEst

    MrEstUnc: str = "MrEstUnc"
    abs_mag_r_est_unc: str = MrEstUnc

    MrdS: str = "MrdS"
    abs_mag_r_entropy_drop: str = MrdS

    # ##### Absolute magnitude plus extinction, r band

    Qr: str = "Qr"
    abs_mag_ext_r: str = Qr

    QrEst: str = "QrEst"
    abs_mag_ext_r_est: str = QrEst

    QrEstUnc: str = "QrEstUnc"
    abs_mag_ext_r_est_unc: str = QrEstUnc

    # TODO: I don't see these in the glossary
    chi2min: str = "chi2min"
    glat: str = "glat"
    glon: str = "glon"
    gr: str = "gr"
    rmag: str = "rmag"


map = ColumnMap()
