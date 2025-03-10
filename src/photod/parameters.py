from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from photod.priors import readPriors


@dataclass
class GlobalParams:
    """Holds fixed parameters"""

    fitColors: tuple
    locusData: np.ndarray
    ArGridList: dict
    locus3DList: dict
    xLabel: str = "FeH"
    yLabel: str = "Mr"
    MrColumn: str = "Mr"

    def __post_init__(self):
        self.Ar1d = self.ArGridList["ArLarge"]
        self._extractMrAndFeH()
        locusColors3d = self.locus3DList["ArLarge"]  # Currently fixed to the large resolution
        # Stack all locus data by colors
        self.locusColors = np.stack([locusColors3d[color] for color in self.fitColors], axis=-1)
        # Create the Qr grid (Qr = Mr + Ar)
        Mr, Ar = jnp.meshgrid(self.Mr1d, self.Ar1d)
        Qr = jnp.round(Mr+Ar, 3)       
        self.QrGrid, self.QrIndices = jnp.unique(Qr, return_inverse=True)


    def _extractMrAndFeH(self):
        # Mr and FeH 1-D grid properties extracted from locus data (same for all stars)
        FeHGrid = self.locusData[self.xLabel]
        MrGrid = self.locusData[self.yLabel]
        FeH1d = np.sort(np.unique(FeHGrid))
        Mr1d = np.sort(np.unique(MrGrid))
        dFeH = FeH1d[1] - FeH1d[0]
        dMr = Mr1d[1] - Mr1d[0]
        self.FeH1d = FeH1d
        self.Mr1d = Mr1d
        self.dFeH = dFeH
        self.dMr = dMr

    def getArgs(self):
        """Arguments to run the calculations for each star"""
        return (self.locusColors, self.Ar1d, self.FeH1d, self.Mr1d, self.dFeH, self.dMr, self.QrGrid, self.QrIndices)

    def getPlottingArgs(self):
        """Arguments to perform plotting for each star"""
        # metadata for plotting likelihood maps below
        FeHmin = np.min(self.FeH1d)
        FeHmax = np.max(self.FeH1d)
        FeHNpts = self.FeH1d.size
        MrFaint = np.max(self.Mr1d)
        MrBright = np.min(self.Mr1d)
        MrNpts = self.Mr1d.size
        print("Mr1d=", np.min(self.Mr1d), np.max(self.Mr1d), len(self.Mr1d))
        print("MrBright, MrFaint=", MrBright, MrFaint)
        mdLocus = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
        return (mdLocus, self.xLabel, self.yLabel, self.Mr1d, self.FeH1d, self.Ar1d)
