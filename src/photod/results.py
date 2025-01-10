from dataclasses import dataclass, field

import numpy as np


@dataclass
class BayesResults:
    """Holds the Bayes results for all stars"""

    chi2min: np.ndarray
    statistics: dict

    # Additional (optional) information
    likeCube: np.ndarray | None = None
    priorCube: np.ndarray | None = None
    postCube: np.ndarray | None = None
    margpostMr: dict | None = None
    margpostFeH: dict | None = None
    margpostAr: dict | None = None
