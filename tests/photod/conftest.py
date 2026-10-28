from pathlib import Path

import lsdb
import pytest
from wd_loci import script, singleMassLoci

TEST_DIR = Path(__file__).parent.parent


@pytest.fixture
def test_data_dir():
    """Directory with the test data."""
    return Path(TEST_DIR) / "data"


@pytest.fixture
def s82_0_5_df(s82_0_5_dir):
    """Stripe 82 test catalog, HEALPix (5, 0)."""
    return lsdb.read_hats(s82_0_5_dir).compute()


@pytest.fixture
def s82_0_5_dir(test_data_dir):
    """Stripe 82 test catalog in HATS format."""
    return test_data_dir / "s82_0_5"


@pytest.fixture
def s82_priors_dir(test_data_dir):
    """Stripe 82 prior maps in HATS format."""
    return test_data_dir / "s82_priors"


@pytest.fixture
def locus_file_path():
    """SDSS/LSST stellar locus table, the one that ships in data/ rather than a second copy of it."""
    return TEST_DIR.parent / "data" / "MSandRGBcolors_v1.3.txt"


@pytest.fixture(scope="module")
def wdLoci(tmp_path_factory):
    """The stars with white dwarf tracks of one mass, and the tracks alone, written to a scratch directory."""
    return singleMassLoci(tmp_path_factory.mktemp("loci"))


@pytest.fixture(scope="module")
def singleLocus(tmp_path_factory):
    """The locus with white dwarf tracks of one mass, for comparison."""
    return singleMassLoci(tmp_path_factory.mktemp("single"))[0]


@pytest.fixture(scope="module")
def massLocus(tmp_path_factory):
    """The files of scripts/make_wd_mass_locus.py, written to a scratch directory, and its mass grid."""
    out = tmp_path_factory.mktemp("massloci")
    grid = script("make_wd_mass_locus").build(out)
    return out / "LSSTlocus_10Gyr_WDmass.txt", grid


@pytest.fixture(scope="module")
def qsoLoci(tmp_path_factory):
    """The mass-resolved white dwarf locus with and without the QSO track, and the mass grid of the second."""
    out = tmp_path_factory.mktemp("qsoloci")
    script("make_wd_mass_locus").build(out)
    grid = script("make_qso_locus").build(out)
    return out / "LSSTlocus_10Gyr_WDmass.txt", out / "LSSTlocus_10Gyr_WDmass_QSO.txt", grid
