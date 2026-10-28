from dataclasses import dataclass

import numpy as np

from photod.locus import (
    CLASS_SEGMENTS,
    EXTRAGALACTIC_SEGMENTS,
    QSO_TLOC,
    SEGMENT_HELIUM_WD,
    SEGMENT_QSO,
    WHITE_DWARF_SEGMENTS,
    extcoeff,
    locusPlateau,
    qsoRedshift,
    segmentAxis,
)


def tLocCellWidths(t, segment):
    """Width of the cell each sorted grid value stands for, within its own segment of the axis.

    Inside a segment it is the mean of the steps either side, and at either end of a segment the one step
    inside it, so that the stretch between two segments belongs to neither and a segment of n points with
    one step h is n h wide. A segment that is a single point has no width to give and is refused.
    """
    t, segment = np.asarray(t, dtype=float), np.asarray(segment)
    d = np.diff(t)
    same = segment[1:] == segment[:-1]
    left = np.concatenate([[np.nan], np.where(same, d, np.nan)])
    right = np.concatenate([np.where(same, d, np.nan), [np.nan]])
    if np.any(np.isnan(left) & np.isnan(right)):
        raise ValueError("a segment of the locus holds a single tLoc value, which has no cell width")
    return np.where(np.isnan(left), right, np.where(np.isnan(right), left, 0.5 * (left + right)))


# The segments whose A_r prior follows the dust in front of them when ArCurveMode is "band": the white dwarfs,
# which are near enough to lie inside the dust layer, and the QSOs, which lie behind all of it.
BAND_SEGMENTS = (*WHITE_DWARF_SEGMENTS, *EXTRAGALACTIC_SEGMENTS)
AR_CURVE_MODES = ("gauss", "band")
# The class the stellar estimates are of (GlobalParams.estimateClass): None for the most probable class of
# star, or one class of locus.CLASS_SEGMENTS for that class always.
ESTIMATE_CLASSES = (None, "MS")


@dataclass
class GlobalParams:
    """Fixed inputs of the Bayes fit: locus grid, color model, A_r grid and the tables derived from them.

    Parameters
    ----------
    fitColors : tuple of str
        Colors used in the fit, e.g. ("ug", "gr", "ri", "iz", "zy").
    locusData : astropy.table.Table
        Locus on a regular grid of xLabel and yLabel (see locus.subsampleLocusData).
    ArGridList : dict
        A_r grids by name, such as locus.getArGridList() returns; ArGridRange picks one of them.
    locus3DList : dict, optional
        The reddened color models of locus.get3DmodelList, which older calls pass. The fit does not use them;
        when given, they are checked against the locus colors reddened with locus.extcoeff().
    xLabel, yLabel : str
        Grid columns of locusData: [Fe/H] and Mr, or [Fe/H] and tLoc.
    MrColumn : str
        Locus column along which the prior maps are tabulated.
    ArGridRange : str
        "Small", "Medium", "Large" or "Fixed".
    computeMrTrue : bool
        For a grid in tLoc, also report quantiles of the true Mr (Mr = tLoc fainter than the turn-off,
        tLoc > turnoffTLoc, and the Mr of the locus table at the turn-off and brighter). Qr = Mr + A_r is
        then computed from the true Mr as well.
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
        zeros, or a star without a dust map, falls back to the flat prior.
    ArCurveMu : ndarray, optional
        Distance moduli of the ArCurves columns.
    ArCurveIndexColumn : str, optional
        Catalog column with the row of ArCurves for each star.
    ArCurveFrac, ArCurveFloor : float
        Width of the prior, sqrt((ArCurveFrac * A_r)^2 + ArCurveFloor^2), for the uncertainty of the 3D map.
    ArCurveMode : str
        "gauss" for that Gaussian on every row. "band" instead keeps the box on every row but those of the
        segments ArCurveRows, where A_r is flat between C(mu) / ArPriorScale - ArPriorOffset and
        ArPriorScale * C(mu) + ArPriorOffset, C(mu) = curve * A_r(map) being the column in front of the
        point's distance; a point of an extragalactic segment is behind all of it, so C = A_r(map). On every
        row the prior is a density in A_r, normalised over its own allowed cells. The box is the band's limit
        far behind the dust, and the curves are an assumed geometry times the 2D column
        (scripts/make_expo_dust_curves.py), not a 3D measurement.
    ArCurveRows : tuple of int
        The segments that take the band, BAND_SEGMENTS by default.
    qcThresholds : ndarray, optional
        The chi2min thresholds of photod.quality.readQcTable, by class, |b| class and colour pattern, from
        which the fit sets FLAG_POOR_FIT_P; without them that flag is never set.
    colorErrFloor : float
        Added in quadrature to every colour error before the fit. The locus is not exact, so with the
        catalog errors alone the posteriors of bright stars are too narrow; 0.03 mag is right for Rubin DP2.
    wdMasses : ndarray, optional
        On a locus whose white dwarf rows are a mass axis (scripts/make_wd_mass_locus.py), the mass of each
        [Fe/H] row in Msun, from which the fit reports WD_mass; None on any other locus.
    estimateClass : str, optional
        The class the stellar estimates (Mr, [Fe/H], A_r, the distance modulus, their spread and the
        multimodality) are of. None takes the most probable of the stars and the white dwarfs, and leaves the
        distance modulus empty where the QSO class wins. "MS" takes the stars for every object, so that no
        star loses its distance to another class; the white dwarf distance and the QSO redshift are then the
        only estimates of those classes. With a class named, the posterior of every class is scaled to its own
        peak before the class masses are taken, so that the estimates of a class far less probable than
        another do not underflow; the masses, and the class probabilities, are then combined in logarithms.
    """

    fitColors: tuple
    locusData: object
    ArGridList: dict
    locus3DList: dict = None
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
    ArCurveMode: str = "gauss"
    ArCurveRows: tuple = BAND_SEGMENTS
    qcThresholds: np.ndarray = None
    wdMasses: np.ndarray = None
    estimateClass: str = None

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

        # the color model is linear in A_r: colors = locusColors2d + A_r * reddVector, each color reddened by
        # its name, whatever the column order of the locus table
        C = extcoeff()
        self.reddVector = np.array([C[c[0]] - C[c[1]] for c in self.fitColors])
        self.locusColors2d = np.stack(
            [np.asarray(self.locusData[c], dtype=float) for c in self.fitColors], axis=-1
        )
        if self.locus3DList is not None:
            model = np.stack([self.locus3DList[f"Ar{self.ArGridRange}"][c] for c in self.fitColors], axis=-1)
            linear = self.locusColors2d.reshape(nFeH, nMr, 1, -1) + self.Ar1d[:, None] * self.reddVector
            if not np.allclose(model, linear, rtol=0, atol=1e-9):
                raise ValueError("locus3DList must hold the locus colors reddened with locus.extcoeff()")
        if nAr > 2 and not np.allclose(np.diff(self.Ar1d), self.dAr, rtol=1e-9, atol=0):
            raise ValueError("the A_r grid must be uniform")
        # the width of the 3D dust prior is sqrt((ArCurveFrac A_r)^2 + ArCurveFloor^2), and the map puts no
        # extinction at all in front of the nearest stars, so without a floor their prior has zero width
        if self.ArCurves is not None and not self.ArCurveFloor > 0:
            raise ValueError("ArCurveFloor must be positive: it is the width of the prior where A_r is zero")
        if self.wdMasses is not None and np.shape(self.wdMasses) != (nFeH,):
            raise ValueError(
                f"wdMasses must give one mass per [Fe/H] row, {nFeH}, not {np.shape(self.wdMasses)}"
            )

        # grid points that only pad an isochrone to the rectangular grid get no prior weight
        self.locusValid = ~locusPlateau(self.locusData, self.xLabel, self.yLabel)
        # What each point of the axis models (locus.SEGMENT_*), and the classes as masks along the axis.
        self.segment1d = segmentAxis(self.locusData, self.xLabel, self.yLabel)
        self.segmentFlat = np.broadcast_to(self.segment1d, (nFeH, nMr)).ravel()
        # every class, present in this locus or not, so that the class columns of a fit do not depend on it
        self.classAxisMasks = {name: np.isin(self.segment1d, codes) for name, codes in CLASS_SEGMENTS.items()}
        self.heliumAxisMask = self.segment1d == SEGMENT_HELIUM_WD
        if self.estimateClass not in ESTIMATE_CLASSES:
            raise ValueError(f"estimateClass must be one of {ESTIMATE_CLASSES}, not {self.estimateClass!r}")
        if self.ArCurveMode not in AR_CURVE_MODES:
            raise ValueError(f"ArCurveMode must be one of {AR_CURVE_MODES}, not {self.ArCurveMode!r}")
        if self.ArCurveMode == "band" and (self.ArCurves is None or nAr < 2):
            raise ValueError("ArCurveMode 'band' needs dust curves and an A_r grid of more than one value")
        self.bandFlat = np.isin(self.segmentFlat, self.ArCurveRows) if self.ArCurveMode == "band" else None
        self.extragalacticFlat = np.isin(self.segmentFlat, EXTRAGALACTIC_SEGMENTS)
        # The QSO track, and the redshift of each of its points, from which the fit reads photoz. Its tLoc
        # stands for a redshift by locus.QSO_TLOC, so a track laid out otherwise is refused rather than read
        # as other redshifts.
        self.qsoAxisIndex = np.where(self.segment1d == SEGMENT_QSO)[0]
        if self.qsoAxisIndex.size and not np.allclose(
            self.Mr1d[self.qsoAxisIndex[[0, -1]]], QSO_TLOC, rtol=0, atol=1e-6
        ):
            raise ValueError(f"the QSO track of the locus must run over tLoc {QSO_TLOC}")
        self.qsoRedshift1d = qsoRedshift(self.Mr1d[self.qsoAxisIndex])
        # Only stars and white dwarfs have an absolute magnitude, so everything built on Mr leaves the QSO
        # points out: put on the Qr grid, their placeholder Mr would stretch it to r + 30.
        stellar = ~self.classAxisMasks["QSO"]
        # The prior maps are densities per unit of [Fe/H] and of the axis, and the fit sums prior times
        # likelihood over the grid points, so each point carries the area of its cell. The [Fe/H] step is the
        # same everywhere; the axis step is not (0.005 on the white dwarf tracks, 0.01 on the stars), and
        # without the widths a track sampled twice as finely gets twice the prior mass it should.
        self.priorWeights = (
            np.broadcast_to(tLocCellWidths(self.Mr1d, self.segment1d), (nFeH, nMr)).ravel() * self.dFeH
        )

        if self.computeMrTrue:
            if self.MrTrueTable is None:
                self.MrTrueTable = np.asarray(self.locusData[self.trueMrLabel], dtype=float).reshape(
                    nFeH, nMr
                )
            if self.MrTrueTable.shape != (nFeH, nMr):
                raise ValueError(f"MrTrueTable must have shape {(nFeH, nMr)}, not {self.MrTrueTable.shape}")
            MrTrue = np.round(np.where(self.Mr1d > self.turnoffTLoc, self.Mr1d, self.MrTrueTable), 3)
            self.MrTrueGrid, MrTrueIndices = np.unique(MrTrue[:, stellar], return_inverse=True)
            # a QSO point adds to the first value, with the zero weight the fit gives it outside its class
            self.MrTrueIndices = np.zeros((nFeH, nMr), dtype=np.int64)
            self.MrTrueIndices[:, stellar] = MrTrueIndices.reshape(nFeH, -1)
        else:
            # Qr and the distance modulus are built from MrTrue, so on a grid that is not the absolute
            # magnitude they would be the reddened tLoc and a distance wrong by the difference, which on the
            # DP2 locus reaches five magnitudes brighter than the turn-off and looks like any other answer
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

        self.QrGrid, stellarIndices = np.unique(
            np.round(MrTrue[:, stellar, None] + self.Ar1d, 3), return_inverse=True
        )
        QrIndices = np.zeros((nFeH, nMr, nAr), dtype=np.int64)
        QrIndices[:, stellar, :] = stellarIndices.reshape(nFeH, -1, nAr)
        independent = np.all(QrIndices == QrIndices[:1], axis=(0, 2)) & stellar
        self.QrColsIndep = np.where(independent)[0]
        self.QrColsDep = np.where(~independent & stellar)[0]
        self.QrIdxIndep = QrIndices[0, independent, :]
        self.QrIdxDep = QrIndices[:, ~independent & stellar, :]
        # the same for the white dwarf columns alone, of which the fit reports a distance of their own
        dependentWD = ~independent & self.classAxisMasks["WD"]
        self.QrColsDepWD = np.where(dependentWD)[0]
        self.QrIdxDepWD = QrIndices[:, dependentWD, :]

    def starArgs(self, nAr=None):
        """Arrays used by the per-star computation, with the A_r grid cut to its first nAr values."""
        nAr = self.Ar1d.size if nAr is None else nAr
        return {
            "classAxisMasks": self.classAxisMasks,
            "heliumAxisMask": self.heliumAxisMask,
            "qsoAxisIndex": self.qsoAxisIndex,
            "qsoRedshift1d": self.qsoRedshift1d,
            "QrColsDepWD": self.QrColsDepWD,
            "QrIdxDepWD": self.QrIdxDepWD[:, :, :nAr],
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
            "ArPriorScale": self.ArPriorScale,
            "ArPriorOffset": self.ArPriorOffset,
            "ArBandRows": self.bandFlat,
            "extragalactic": self.extragalacticFlat,
            "estimateMask": None if self.estimateClass is None else self.classAxisMasks[self.estimateClass],
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
        # the true absolute magnitude of every grid point, the one the fit builds Qr from: with the locus
        # parametrised by tLoc the second grid axis is not a magnitude (see plotting.getQmap)
        MrTrue = self.MrTrueFlat.reshape(self.FeH1d.size, self.Mr1d.size)
        return (mdLocus, self.xLabel, self.yLabel, self.Mr1d, self.FeH1d, self.Ar1d, MrTrue)
