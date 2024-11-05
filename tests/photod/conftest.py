from pathlib import Path

import lsdb
import numpy as np
import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent.parent


@pytest.fixture
def test_data_dir():
    return Path(TEST_DIR) / "data"

@pytest.fixture
def s82_0_5_df(test_data_dir):
    return lsdb.read_hats(test_data_dir / "s82_0_5").compute()
