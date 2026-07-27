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
    ArGridRange: str = 'Large'

    # === NEW: toggle for tLoc -> Mr_true recalculation ===
    computeMrTrue: bool = False
    # Column name in locusData holding the TRUE Mr value (as opposed to
    # yLabel, which may be "tLoc"). Only used when computeMrTrue=True.
    trueMrLabel: str = "Mr"
    # Optional manual override -- normally left None so it's auto-derived
    # from locusData[trueMrLabel]. Only set this yourself if you want to
    # supply a table from somewhere other than locusData.
    MrTrueTable: np.ndarray = None
    # === END NEW ===

    def __post_init__(self):
        self.Ar1d = self.ArGridList["Ar{}".format(self.ArGridRange)]
        self._extractMrAndFeH()
        locusColors3d = self.locus3DList["Ar{}".format(self.ArGridRange)]
        self.locusColors = np.stack([locusColors3d[color] for color in self.fitColors], axis=-1)

        # Create the Qr grid (Qr = Mr + Ar)
        Mr, Ar = jnp.meshgrid(self.Mr1d, self.Ar1d)
        Qr = jnp.round(Mr + Ar, 3)
        self.QrGrid, self.QrIndices = jnp.unique(Qr, return_inverse=True)

        # === NEW: precompute the Mr_true value grid + indices, once ===
        if self.computeMrTrue:
            if self.MrTrueTable is None:
                # auto-derive directly from locusData -- no manual table needed
                self.MrTrueTable = self._extractMrTrueTable()

            expectedShape = (self.FeH1d.size, self.Mr1d.size)
            if self.MrTrueTable.shape != expectedShape:
                raise ValueError(
                    f"MrTrueTable shape {self.MrTrueTable.shape} does not match "
                    f"expected (FeH1d.size, {self.yLabel}1d.size) = {expectedShape}."
                )

            # grid of the yLabel values (tLoc), same shape as MrTrueTable
            FeHGridMesh, yLabelGridMesh = jnp.meshgrid(self.FeH1d, self.Mr1d, indexing="ij")
            # Mr_true = tLoc itself when tLoc > 4, else the true Mr looked up from locusData
            MrTrueRaw = jnp.where(yLabelGridMesh > 4, yLabelGridMesh, jnp.asarray(self.MrTrueTable))
            MrTrueRaw = jnp.round(MrTrueRaw, 3)
            self.MrTrueGrid, self.MrTrueIndices = jnp.unique(MrTrueRaw, return_inverse=True)
        else:
            self.MrTrueGrid = jnp.zeros(1)
            self.MrTrueIndices = jnp.zeros((self.FeH1d.size, self.Mr1d.size), dtype=jnp.int32)
        # === END NEW ===

    def _extractMrAndFeH(self):
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

    # === NEW ===
    def _extractMrTrueTable(self):
        """Build the (FeH, yLabel) -> true Mr lookup table directly from locusData.

        locusData already carries both the grid column (self.yLabel, e.g. "tLoc")
        and the true Mr column (self.trueMrLabel, e.g. "Mr") on the same rows.
        This mirrors the exact reshape used in make3DlocusList
        (LocusNP.reshape(FeH1d.size, Mr1d.size)), so it relies on locusData
        being laid out on that same regular grid.
        """
        nFeH = self.FeH1d.size
        nY = self.Mr1d.size  # size of the yLabel (tLoc) grid
        trueMrGrid = np.asarray(self.locusData[self.trueMrLabel]).reshape(nFeH, nY)
        return trueMrGrid
    # === END NEW ===

    def getArgs(self):
        """Arguments to run the calculations for each star"""
        return (
            self.locusColors,
            self.Ar1d,
            self.FeH1d,
            self.Mr1d,
            self.dFeH,
            self.dMr,
            self.QrGrid,
            self.QrIndices,
            self.MrTrueGrid,      # === NEW ===
            self.MrTrueIndices,   # === NEW ===
        )

    def getPlottingArgs(self):
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