from dataclasses import dataclass

import numpy as np

from photod.locus import extcoeff, locusPlateau


@dataclass
class GlobalParams:
    """Fixed inputs of the Bayes fit: locus grid, 3D color model, A_r grid and the tables derived from them.

    Parameters
    ----------
    fitColors : tuple of str
        Colors used in the fit, e.g. ("ug", "gr", "ri", "iz", "zy").
    locusData : astropy.table.Table
        Locus on a regular grid of xLabel and yLabel (see locus.subsampleLocusData).
    ArGridList, locus3DList : dict
        A_r grids and 3D color models from locus.get3DmodelList.
    xLabel, yLabel : str
        Grid columns of locusData: [Fe/H] and Mr, or [Fe/H] and tLoc.
    MrColumn : str
        Locus column along which the prior maps are tabulated.
    ArGridRange : str
        "Small", "Medium", "Large" or "Fixed".
    computeMrTrue : bool
        For a grid in tLoc, also report quantiles of the true Mr (Mr = tLoc above the turn-off, and the Mr of
        the locus table below it). Qr = Mr + A_r is then computed from the true Mr as well.
    trueMrLabel : str
        Locus column with the true Mr.
    MrTrueTable : ndarray, optional
        True Mr on the (xLabel, yLabel) grid; by default locusData[trueMrLabel].
    turnoffTLoc : float
        Turn-off point of the tLoc parametrization.
    ArMapColumn : str, optional
        Catalog column with the dust-map A_r. The A_r prior is flat between 0 and
        ArPriorScale * A_r(map) + ArPriorOffset; without a dust map it is flat over the whole A_r grid.
    ArCurves : ndarray, optional
        Shapes A_r(mu) / A_r(total) of a 3D dust map, one row per sightline, on the grid ArCurveMu. With them
        the A_r prior is no longer flat: locus point i puts the star at mu = r - Mr_i - A_r, so the map gives
        the extinction consistent with that distance and the prior becomes a Gaussian around it. A row of
        zeros, or a star without a dust map, falls back to the flat prior. See scripts/make_dust_curves.py.
    ArCurveMu : ndarray, optional
        Distance moduli of the ArCurves columns.
    ArCurveIndexColumn : str, optional
        Catalog column with the row of ArCurves for each star.
    ArCurveFrac, ArCurveFloor : float
        Width of the prior, sqrt((ArCurveFrac * A_r)^2 + ArCurveFloor^2), for the uncertainty of the 3D map.
    colorErrFloor : float
        Added in quadrature to every colour error before the fit. The locus is not exact, so with the
        catalog errors alone the posteriors of bright stars are too narrow; 0.03 mag is right for Rubin DP2.
    """

    fitColors: tuple
    locusData: object
    ArGridList: dict
    locus3DList: dict
    xLabel: str = "FeH"
    yLabel: str = "Mr"
    MrColumn: str = "Mr"
    ArGridRange: str = "Large"
    computeMrTrue: bool = False
    trueMrLabel: str = "Mr"
    MrTrueTable: np.ndarray = None
    turnoffTLoc: float = 4.0
    ArMapColumn: str = None
    ArPriorScale: float = 1.3
    ArPriorOffset: float = 0.1
    colorErrFloor: float = 0.0
    ArCurves: np.ndarray = None
    ArCurveMu: np.ndarray = None
    ArCurveIndexColumn: str = None
    ArCurveFrac: float = 0.15
    ArCurveFloor: float = 0.05

    def __post_init__(self):
        self.FeH1d = np.unique(np.asarray(self.locusData[self.xLabel], dtype=float))
        self.Mr1d = np.unique(np.asarray(self.locusData[self.yLabel], dtype=float))
        self.dFeH = self.FeH1d[1] - self.FeH1d[0]
        self.dMr = self.Mr1d[1] - self.Mr1d[0]
        self.Ar1d = np.atleast_1d(np.asarray(self.ArGridList[f"Ar{self.ArGridRange}"], dtype=float))
        self.dAr = self.Ar1d[1] - self.Ar1d[0] if self.Ar1d.size > 1 else 0.01
        nFeH, nMr, nAr = self.FeH1d.size, self.Mr1d.size, self.Ar1d.size

        # Everything below reshapes the locus table as it stands, so its rows have to be the rectangular grid
        # of FeH1d by Mr1d in that order. A table holding the same values in another order builds a model
        # that is mirrored in one axis or the other and answers with it, saying nothing.
        if len(self.locusData) != nFeH * nMr:
            raise ValueError(
                f"the locus is not a rectangular grid: {len(self.locusData)} rows for {nFeH} x {nMr} "
                f"values of {self.xLabel} and {self.yLabel}"
            )
        onGrid = [
            np.asarray(self.locusData[label], dtype=float).reshape(nFeH, nMr)
            for label in (self.xLabel, self.yLabel)
        ]
        if not (np.allclose(onGrid[0], self.FeH1d[:, None]) and np.allclose(onGrid[1], self.Mr1d[None, :])):
            raise ValueError(
                f"the locus rows must run over {self.yLabel} within blocks of increasing {self.xLabel}, "
                "both ascending"
            )

        # the color model is linear in A_r: colors = locusColors2d + A_r * reddVector
        C = extcoeff()
        self.reddVector = np.array([C[c[0]] - C[c[1]] for c in self.fitColors])
        model = np.stack([self.locus3DList[f"Ar{self.ArGridRange}"][c] for c in self.fitColors], axis=-1)
        self.locusColors2d = (model[:, :, 0, :] - self.Ar1d[0] * self.reddVector).reshape(nFeH * nMr, -1)
        linear = self.locusColors2d.reshape(nFeH, nMr, 1, -1) + self.Ar1d[:, None] * self.reddVector
        if not np.allclose(model, linear, rtol=0, atol=1e-9):
            raise ValueError("locus3DList must hold the locus colors reddened with locus.extcoeff()")
        if nAr > 2 and not np.allclose(np.diff(self.Ar1d), self.dAr, rtol=1e-9, atol=0):
            raise ValueError("the A_r grid must be uniform")
        # the width of the 3D dust prior is sqrt((ArCurveFrac A_r)^2 + ArCurveFloor^2), and the map puts no
        # extinction at all in front of the nearest stars, so without a floor their prior has zero width
        if self.ArCurves is not None and not self.ArCurveFloor > 0:
            raise ValueError("ArCurveFloor must be positive: it is the width of the prior where A_r is zero")

        # grid points that only pad an isochrone to the rectangular grid get no prior weight
        self.locusValid = ~locusPlateau(self.locusData, self.xLabel, self.yLabel)

        if self.computeMrTrue:
            if self.MrTrueTable is None:
                self.MrTrueTable = np.asarray(self.locusData[self.trueMrLabel], dtype=float).reshape(
                    nFeH, nMr
                )
            if self.MrTrueTable.shape != (nFeH, nMr):
                raise ValueError(f"MrTrueTable must have shape {(nFeH, nMr)}, not {self.MrTrueTable.shape}")
            MrTrue = np.round(np.where(self.Mr1d > self.turnoffTLoc, self.Mr1d, self.MrTrueTable), 3)
            self.MrTrueGrid, MrTrueIndices = np.unique(MrTrue, return_inverse=True)
            self.MrTrueIndices = MrTrueIndices.reshape(nFeH, nMr)
        else:
            # Qr and the distance modulus are built from MrTrue, so on a grid that is not the absolute
            # magnitude they would be the reddened tLoc and a distance wrong by the difference, which on the
            # DP2 locus reaches five magnitudes above the turn-off and looks like any other answer
            if self.trueMrLabel in self.locusData.colnames and self.yLabel != self.trueMrLabel:
                raise ValueError(
                    f"the locus is on a {self.yLabel} grid, so computeMrTrue=True is needed for Qr and the "
                    f"distance modulus to come from {self.trueMrLabel} rather than from {self.yLabel}"
                )
            MrTrue = np.broadcast_to(self.Mr1d, (nFeH, nMr))
            self.MrTrueGrid, self.MrTrueIndices = None, None

        # Qr = Mr_true + A_r on the (FeH, Mr, Ar) grid. Columns where the index does not depend on [Fe/H]
        # are summed over [Fe/H] before the Qr histogram is filled.
        self.MrTrueFlat = np.asarray(MrTrue, dtype=float).reshape(-1)

        self.QrGrid, QrIndices = np.unique(np.round(MrTrue[:, :, None] + self.Ar1d, 3), return_inverse=True)
        QrIndices = QrIndices.reshape(nFeH, nMr, nAr)
        independent = np.all(QrIndices == QrIndices[:1], axis=(0, 2))
        self.QrColsIndep = np.where(independent)[0]
        self.QrColsDep = np.where(~independent)[0]
        self.QrIdxIndep = QrIndices[0, independent, :]
        self.QrIdxDep = QrIndices[:, ~independent, :]

    def starArgs(self, nAr=None):
        """Arrays used by the per-star computation, with the A_r grid cut to its first nAr values."""
        nAr = self.Ar1d.size if nAr is None else nAr
        return {
            "locusColors2d": self.locusColors2d,
            "reddVector": self.reddVector,
            "ArFull": self.Ar1d,
            "Ar1d": self.Ar1d[:nAr],
            "dAr": self.dAr,
            "FeH1d": self.FeH1d,
            "Mr1d": self.Mr1d,
            "dFeH": self.dFeH,
            "dMr": self.dMr,
            "QrGrid": self.QrGrid,
            "QrColsIndep": self.QrColsIndep,
            "QrColsDep": self.QrColsDep,
            "QrIdxIndep": self.QrIdxIndep[:, :nAr],
            "QrIdxDep": self.QrIdxDep[:, :, :nAr],
            "MrTrueGrid": self.MrTrueGrid,
            "MrTrueIndices": self.MrTrueIndices,
            "MrTrueFlat": self.MrTrueFlat,
            "ArCurves": self.ArCurves,
            "ArCurveMu": self.ArCurveMu,
            "ArCurveFrac": self.ArCurveFrac,
            "ArCurveFloor": self.ArCurveFloor,
        }

    def getPlottingArgs(self):
        """Locus metadata and grids in the form used by the plotting functions."""
        mdLocus = np.array(
            [
                self.FeH1d.min(),
                self.FeH1d.max(),
                self.FeH1d.size,
                self.Mr1d.max(),
                self.Mr1d.min(),
                self.Mr1d.size,
            ]
        )
        return (mdLocus, self.xLabel, self.yLabel, self.Mr1d, self.FeH1d, self.Ar1d)
