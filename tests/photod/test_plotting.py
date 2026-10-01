import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from test_bayes import COLORS, brute_force, make_locus, make_params, make_stars

import photod.plotting as pt
from photod.priors import getBayesConstants, getPriorMapIndex
from photod.results import BayesResults
from photod.stats import getStats

# nothing is displayed or rendered here: the tests read back what the plots were given, not their pixels
matplotlib.use("Agg")

# stars of make_stars(n=4) whose posterior sits below the turn-off of the tLoc grid, where the true absolute
# magnitude is several magnitudes brighter than tLoc, and above it, where the two are the same
GIANT, DWARF = 0, 1
# an A_r grid of the length and step a real run uses (scripts/run_dp2.py), rather than the test locus grid
RUN_AR_GRID = np.arange(0, 5.0 + 1e-9, 0.02)


@pytest.fixture(autouse=True)
def plainFigures(monkeypatch):
    """Figures at a low resolution, closed again afterwards: the tests read them back, they do not look."""
    monkeypatch.setitem(matplotlib.rcParams, "figure.dpi", 20)
    monkeypatch.setitem(matplotlib.rcParams, "savefig.dpi", 20)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def run():
    """A tLoc locus with a few stars, the posterior cube of each, and the prior map they were fitted with.

    The cubes are built here with numpy rather than by bayes.makeBayesEstimates3d, which compiles a kernel
    for them and on its own takes longer than everything in this file put together. They are the cubes of
    test_bayes.brute_force, which test_bayes.test_estimates_match_brute_force pins to the fit to 1e-9.
    """
    locus = make_locus(True)
    params = make_params(locus, True)
    stars = make_stars(locus, n=4)
    priorGrid = np.ones((getBayesConstants()["rmagNsteps"], len(locus)))
    cubes = [posteriorCube(stars.iloc[i], params, priorGrid) for i in range(len(stars))]
    return params, stars, priorGrid, cubes


def posteriorCube(star, params, priorGrid):
    """The (FeH, Mr, Ar) posterior cube of one star, written out as test_bayes.brute_force writes it."""
    model = np.stack([params.locus3DList[f"Ar{params.ArGridRange}"][c] for c in COLORS], axis=-1)
    obs = star[list(COLORS)].to_numpy(float)
    err = star[[color + "Err" for color in COLORS]].to_numpy(float)
    chi2 = np.sum(((obs - model) / err) ** 2, axis=-1)
    allowed = params.ArPriorScale * star["ArMap"] + params.ArPriorOffset >= params.Ar1d
    prior = priorGrid[getPriorMapIndex(star["rmag"])].reshape(params.FeH1d.size, params.Mr1d.size)
    return prior[:, :, None] * np.exp(-0.5 * (chi2 - chi2.min())) * allowed


def marginals(cube, params):
    """The prior, likelihood and posterior marginals plotStar takes, in its order and all from one cube."""
    steps = (((0, 1), params.dAr), ((0, 2), params.dMr), ((1, 2), params.dFeH))
    return [{k: cube.sum(axis=axes) / cube.sum() / step for k in range(3)} for axes, step in steps]


def median(x, pdf):
    """The median of a sampled distribution, on the cumulative distribution stats.py interpolates."""
    cdf = (np.cumsum(pdf) - 0.5 * pdf) / np.sum(pdf)
    return float(np.interp(0.5, cdf, x))


def rowCoordinates(image):
    """The coordinate of each row of a drawn image, from the extent and the origin it was drawn with."""
    bottom, top = image.get_extent()[2:]
    nRows = np.asarray(image.get_array()).shape[0]
    edges = np.linspace(*((bottom, top) if image.origin == "lower" else (top, bottom)), nRows + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def openFigures():
    """The figures the plots left open."""
    return [plt.figure(number) for number in plt.get_fignums()]


def draw(figure):
    """Render a figure, which is when matplotlib checks the range of a LogNorm, on a postage stamp."""
    figure.set_size_inches(1, 1)
    figure.canvas.draw()


def drawnPanels():
    """Every panel of every open figure that something was drawn on, colorbars and empty corners aside."""
    return [ax for fig in openFigures() for ax in fig.axes if ax.images or ax.lines]


def drawnImages():
    """Every image of every open figure, with the axis labels of the panel it was drawn on."""
    return [
        (ax.get_xlabel(), ax.get_ylabel(), ax.images[0])
        for fig in openFigures()
        for ax in fig.axes
        if ax.images
    ]


def markedValues():
    """The values marked as true on each 1D panel of the open figures, by panel label.

    A true value is marked by a black vertical line; the red one is the mean of the posterior.
    """
    marked = {}
    for fig in openFigures():
        for ax in fig.axes:
            for line in ax.lines:
                x = line.get_xdata()
                if len(x) == 2 and x[0] == x[1] and line.get_color() == "k":
                    marked.setdefault(ax.get_ylabel(), set()).add(float(x[0]))
    return marked


@pytest.mark.parametrize("star", [GIANT, DWARF])
def test_qr_is_the_qr_of_the_fit(run, star):
    """The Qr of the plots is the Qr of the fit: Mr + A_r built from the true Mr of each grid point.

    On a tLoc grid the second axis of the cube is not a magnitude, and a Qr read off that axis is wrong by
    the difference everywhere below the turn-off, which is where the giant of the two stars sits.
    """
    params, stars, priorGrid, cubes = run
    MrTrue = params.getPlottingArgs()[-1]
    expected = brute_force(stars.iloc[star], params, priorGrid)["Qr_quantile_median"]

    Qr1d, margQr = pt.showQrCornerPlot(
        cubes[star], params.Mr1d, params.FeH1d, params.Ar1d, logScale=True, MrTrue=MrTrue
    )
    dQr = Qr1d[1] - Qr1d[0]
    assert_allclose(median(Qr1d, np.asarray(margQr)), expected, atol=dQr)

    # the mean and the width plotStar reports come from that same marginal
    QrEst, QrEstUnc = getStats(Qr1d, margQr)
    assert abs(float(QrEst) - expected) < max(0.5, float(QrEstUnc))

    # the second axis of the cube taken for a magnitude is what the giant's Qr must not be read off
    QmapAxis, QrAxis = pt.getQmap(cubes[star], params.FeH1d, params.Mr1d, params.Ar1d)
    offAxis = abs(median(QrAxis, QmapAxis.sum(axis=0)) - expected)
    if star == GIANT:
        assert offAxis > 2.0
    else:
        assert offAxis < dQr


@pytest.mark.parametrize("nAr", [None, RUN_AR_GRID.size])
def test_qmap_keeps_all_the_posterior_weight(run, nAr):
    """getQmap projects the cube onto the Qr axis rather than sampling it, so no weight is left behind."""
    params, stars, priorGrid, cubes = run
    MrTrue = params.getPlottingArgs()[-1]
    if nAr is None:
        Ar1d, cube = params.Ar1d, cubes[GIANT]
    else:
        Ar1d = RUN_AR_GRID
        cube = np.random.default_rng(7).random((params.FeH1d.size, params.Mr1d.size, nAr))

    Qmap, Qr1d = pt.getQmap(cube, params.FeH1d, params.Mr1d, Ar1d, MrTrue)
    dQr = Qr1d[1] - Qr1d[0]
    assert_allclose(Qmap.sum(), cube.sum(), rtol=1e-12)
    assert_allclose(np.diff(Qr1d), dQr)
    assert dQr == max(params.dMr, Ar1d[1] - Ar1d[0])

    # and it projects to the right place: the mean of the map is the mean Qr of the cube itself
    Qr = MrTrue[:, :, None] + Ar1d
    assert Qr1d[0] <= Qr.min() and Qr.max() <= Qr1d[-1]
    assert abs((Qmap * Qr1d).sum() / Qmap.sum() - np.average(Qr, weights=cube)) < 0.5 * dQr

    # a single cell of the cube goes to the single Qr bin nearest its own Qr, an exact grid value included
    one = np.zeros(cube.shape)
    one[1, 8, 0] = 1.0
    Qmap, Qr1d = pt.getQmap(one, params.FeH1d, params.Mr1d, Ar1d, MrTrue)
    assert Qmap.sum() == 1.0 and np.count_nonzero(Qmap) == 1
    assert abs(Qr1d[Qmap[1].argmax()] - (MrTrue[1, 8] + Ar1d[0])) <= 0.5 * dQr


def test_the_marginals_are_densities(run):
    """Both marginals of the Qr map integrate to 1, each with the step of its own axis."""
    params, stars, priorGrid, cubes = run
    Qr1d, margQr = pt.showQrCornerPlot(
        cubes[DWARF],
        params.Mr1d,
        params.FeH1d,
        params.Ar1d,
        logScale=True,
        MrTrue=params.getPlottingArgs()[-1],
    )
    assert_allclose(np.sum(margQr) * (Qr1d[1] - Qr1d[0]), 1.0, rtol=1e-5)

    # as plotted, which is where a step handed to the wrong marginal would show
    panels = {ax.get_ylabel(): ax for fig in openFigures() for ax in fig.axes}
    for label, step in (("p(Qr)", Qr1d[1] - Qr1d[0]), ("p(FeH)", params.dFeH)):
        assert_allclose(np.sum(panels[label].lines[0].get_ydata()) * step, 1.0, rtol=1e-5)


@pytest.mark.parametrize("logScale", [True, False])
def test_the_maps_are_drawn_the_right_way_up(run, logScale):
    """Every row of a map is drawn at the coordinate of the grid value it holds, on a log scale as well."""
    params, stars, priorGrid, cubes = run
    mdLocus = params.getPlottingArgs()[0]
    grids = {"tLoc": params.Mr1d, "Ar": params.Ar1d}
    pt.showCornerPlot3(
        cubes[DWARF], params.Mr1d, params.FeH1d, params.Ar1d, mdLocus, "FeH", "tLoc", logScale=logScale
    )
    images = drawnImages()
    assert sorted(yLabel for _, yLabel, _ in images) == ["Ar", "Ar", "tLoc"]
    for _, yLabel, image in images:
        grid = grids[yLabel]
        assert_allclose(rowCoordinates(image), grid, atol=0.5 * (grid[1] - grid[0]))

    plt.close("all")
    Qr1d, _ = pt.showQrCornerPlot(
        cubes[DWARF],
        params.Mr1d,
        params.FeH1d,
        params.Ar1d,
        logScale=logScale,
        MrTrue=params.getPlottingArgs()[-1],
    )
    ((_, _, image),) = drawnImages()
    assert_allclose(rowCoordinates(image), Qr1d, atol=0.5 * (Qr1d[1] - Qr1d[0]))


def test_a_star_without_true_values(run):
    """A catalog of real stars carries none of the true values, and then nothing is marked as true.

    Through plotStars, which is how a notebook plots a partition of one, so with the arguments the fit
    itself hands the plots.
    """
    params, stars, priorGrid, cubes = run
    cube = cubes[DWARF]
    assert not {"FeH", "Mr", "tLoc", "Ar"} & set(stars.columns)

    result = BayesResults(np.zeros(1), {})
    result.priorCube = result.likeCube = result.postCube = cube[None]
    result.margpostAr, result.margpostMr, result.margpostFeH = (
        {k: value[None] for k, value in marginal.items()} for marginal in marginals(cube, params)
    )
    ((QrEst, QrEstUnc),) = pt.plotStars(stars.iloc[[DWARF]], [result], *params.getPlottingArgs())
    assert np.isfinite(float(QrEst)) and float(QrEstUnc) > 0
    assert not markedValues()
    for ax in drawnPanels():
        # neither a marker nor a line at the sentinel, which would stretch the axis out to it
        assert not ax.collections
        assert min(ax.get_xlim()[0], ax.get_ylim()[0]) > pt.NO_TRUTH / 2

    # the panels are labeled with the grid the locus is on, which here is tLoc rather than Mr
    assert {ax.get_xlabel() for ax in drawnPanels()} & {"tLoc", "Mr"} == {"tLoc"}


def test_true_values_go_on_their_own_axis(run):
    """The true values a catalog does carry are marked, each on the axis it belongs to."""
    params, stars, priorGrid, cubes = run
    cube = cubes[DWARF]
    star = stars.iloc[DWARF].copy()
    # the true Mr of a star below the turn-off is not its tLoc, and the Qr of the fit is made of the Mr
    truth = {"FeH": -1.0, "tLoc": 5.0, "Mr": 2.5, "Ar": 0.1}
    for column, value in truth.items():
        star[column] = value
    axisTruth = {"FeH": -1.0, "tLoc": 5.0, "Ar": 0.1, "Qr = Mr + Ar": 2.6}

    pt.plotStar(star, *marginals(cube, params), cube, cube, cube, *params.getPlottingArgs())
    marked = markedValues()
    for axis, value in axisTruth.items():
        label = f"p({axis.split(' ')[0]})"
        assert any(abs(x - value) < 1e-9 for x in marked[label]), label
    for xLabel, yLabel, image in drawnImages():
        ((x, y),) = image.axes.collections[0].get_offsets()
        assert (x, y) == pytest.approx((axisTruth[xLabel], axisTruth[yLabel]))


def test_an_empty_map_is_drawn_flat():
    """A map that is zero everywhere has no range to put on a log scale, and is drawn without one."""
    mdLocus = [-2.0, 0.0, 2, 10.0, 1.0, 2]
    pt.show3Flat2Dmaps(np.zeros(4), np.ones(4), np.ones(4), mdLocus, "FeH", "Mr", logScale=True)
    draw(openFigures()[0])
    values = np.asarray(drawnImages()[0][2].get_array())
    assert np.isfinite(values).all() and not values.any()


def test_a_faint_map_keeps_the_log_scale_over_it():
    """show2Dmap puts the floor of its log scale below the map rather than at a fixed value above it."""
    Xgrid, Ygrid = np.meshgrid(np.linspace(-2, 0, 5), np.linspace(1, 10, 4))
    Z = np.full(Xgrid.size, 1e-5)
    pt.show2Dmap(Xgrid, Ygrid, Z, [-2.0, 0.0, 5, 10.0, 1.0, 4], "FeH", "Mr", logScale=True)
    figure = openFigures()[0]
    draw(figure)
    low, high = figure.axes[0].images[0].get_clim()
    assert low < Z.max() and high == pytest.approx(Z.max())

    plt.close("all")
    pt.show2Dmap(Xgrid, Ygrid, np.zeros(Xgrid.size), [-2.0, 0.0, 5, 10.0, 1.0, 4], "FeH", "Mr", logScale=True)
    draw(openFigures()[0])


def test_saving_creates_the_directory(tmp_path, monkeypatch):
    """A saved figure goes to plots/ of the working directory, which it creates, under a name of its own."""
    monkeypatch.chdir(tmp_path)
    mdLocus = [-2.0, 0.0, 2, 10.0, 1.0, 2]
    pt.show3Flat2Dmaps(np.ones(4), np.ones(4), np.ones(4), mdLocus, "FeH", "Mr", saveFig=True)
    assert [path.name for path in (tmp_path / "plots").iterdir()] == ["bayesPanels.png"]

    # the name of the run goes into the file name, and nothing goes there when there is no name
    saved = []
    monkeypatch.setattr(plt, "savefig", saved.append)
    pt.show3Flat2Dmaps(np.ones(4), np.ones(4), np.ones(4), mdLocus, "FeH", "Mr", saveFig=True, file_ext="-7")
    assert [path.name for path in saved] == ["bayesPanels-7.png"]
