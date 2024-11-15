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
    priorsRootName: str
    xLabel: str = "FeH"
    yLabel: str = "Mr"
    MrColumn: str = "Mr"

    def __post_init__(self):
        self.Ar1d = self.ArGridList["ArLarge"]
        priorGrid = readPriors(self.priorsRootName, self.locusData, self.MrColumn)
        self.priorGrid = jnp.array(list(priorGrid.values()))

        self._extract_Mr_and_FeH()

        locusColors3d = self.locus3DList["ArLarge"]  # Currently fixed to the large resolution
        # Stack all locus data by colors
        self.locusColors = np.stack([locusColors3d[color] for color in self.fitColors], axis=-1)

    def _extract_Mr_and_FeH(self):
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

    def get_args(self):
        """Arguments to run the calculations for each star"""
        return (self.locusColors, self.Ar1d, self.FeH1d, self.Mr1d, self.priorGrid, self.dFeH, self.dMr)

    def get_plotting_args(self):
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
