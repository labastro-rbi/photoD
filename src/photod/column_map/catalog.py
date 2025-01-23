"""
Column map for catalog data.
"""

from dataclasses import dataclass
from photod.column_map import base


@dataclass
class ColumnMap(base.ColumnMap):

    def __init__(self):
        super().__init__("Catalog column map", "Reference catalog columns")

    # Est suffix = estimate
    # Unc suffix = uncertainty
    # EstUnc = estimated uncertainty as a +/- delta for the
    #     quantity it modifies (confidence interval)
    # dS suffix: drop in entropy

    # ##### Extinction in r band
    extinction_r: str = "Ar"
    extinction_r_est = "ArEst"
    extinction_r_est_unc = "ArEstUnc"

    extinction_r_entropy_drop = "ArdS"

    # ##### Metallicity
    metallicity: str = "FeH"
    metallicity_est: str = "FeHEst"
    metallicity_est_unc: str = "FeHEstUnc"

    metallicity_entropy_drop: str = "FeHdS"

    # ##### Absolute magnitude, r band

    abs_mag_r: str = "Mr"
    abs_mag_r_est: str = "MrEst"
    abs_mag_r_est_unc: str = "MrEstUnc"

    abs_mag_r_entropy_drop: str = "MrdS"

    # ##### Absolute magnitude plus extinction, r band

    abs_mag_ext_r: str = "Qr"
    abs_mag_ext_r_est: str = "QrEst"
    abs_mag_ext_r_est_unc: str = "QrEstUnc"

    observed_mag_r: str = "rmag"

    galactic_latitude: str = "glat"
    galactic_longitude: str = "glon"

    chi_sq_min: str = "chi2min"

    g_minus_r: str = "gr"


m = ColumnMap()
