from pathlib import Path

import lsdb
import pytest

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
def locus_file_path(test_data_dir):
    """SDSS/LSST stellar locus table."""
    return test_data_dir / "locus" / "MSandRGBcolors_v1.3.txt"
