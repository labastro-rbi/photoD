from dataclasses import dataclass, field
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

    # === NEW: toggle + table for tLoc -> Mr_true recalculation ===
    # computeMrTrue is a plain python bool (static, not traced) -- it must be
    # passed separately to jax.jit functions as a static_argname, never inside
    # the getArgs() tuple, or tracing will break on the `if` branches.
    computeMrTrue: bool = False
    # MrTrueTable must be a (FeH1d.size, Mr1d.size) array giving the Mr_true
    # value to use for tLoc <= 4 (looked up on the same grid as postCube).
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
        # This mirrors the Qr pattern exactly. It's built once per run (not
        # per star), so doing this here -- rather than inside the per-star
        # jitted loop -- is what makes it cheap at millions of calls.
        if self.computeMrTrue:
            if self.MrTrueTable is None:
                raise ValueError("computeMrTrue=True requires MrTrueTable to be provided.")
            expectedShape = (self.FeH1d.size, self.Mr1d.size)
            if self.MrTrueTable.shape != expectedShape:
                raise ValueError(
                    f"MrTrueTable shape {self.MrTrueTable.shape} does not match "
                    f"expected (FeH1d.size, Mr1d.size) = {expectedShape}. "
                    "Transpose your table if it's oriented (Mr, FeH) instead."
                )
            FeHGridMesh, MrGridMesh = jnp.meshgrid(self.FeH1d, self.Mr1d, indexing="ij")
            MrTrueRaw = jnp.where(MrGridMesh > 4, MrGridMesh, jnp.asarray(self.MrTrueTable))
            MrTrueRaw = jnp.round(MrTrueRaw, 3)
            self.MrTrueGrid, self.MrTrueIndices = jnp.unique(MrTrueRaw, return_inverse=True)
        else:
            # Harmless placeholders so getArgs() always returns a fixed-shape
            # tuple regardless of the toggle -- these are never touched when
            # computeMrTrue=False, since that's also a static jit argument.
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