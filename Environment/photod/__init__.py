"""Top-level package for photoD."""

__author__ = """Karlo Mrakovcic and Zeljko Ivezic"""
__email__ = "karlo.mrakovcic@uniri.hr"
__version__ = "0.1.0"

import importlib.util
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.path.append("..")

numpy_installed = importlib.util.find_spec("numpy")
tensorflow_installed = importlib.util.find_spec("tensorflow")
if (numpy_installed is None) or (tensorflow_installed is None):
    raise ImportError('Numpy and Tensorflow (v2.11) are required to use photoD.')
else:
    from photod.create_model import PhotoD
    import photod.model_tools

pyplot_installed = importlib.util.find_spec("matplotlib")
scipy_installed = importlib.util.find_spec("scipy")
if (pyplot_installed is None) or (scipy_installed is None):
    logging.warning('No plotting capabilities. Install matplotlib and/or scipy to enable plotting.')
else:
    import photod.plot_tools
