from dataclasses import dataclass

import numpy as np


@dataclass
class BayesResults:
    """Holds the Bayes results for all stars"""

    likeCube: np.ndarray
    priorCube: np.ndarray
    chi2min: np.ndarray
    postCube: np.ndarray
    margpostMr: dict
    margpostFeH: dict
    margpostAr: dict
    statistics: dict

    def get_plotting_args(self):
        """Arguments to perform plotting for each star"""
        return (
            self.margpostAr,
            self.margpostMr,
            self.margpostFeH,
            self.likeCube,
            self.priorCube,
            self.postCube,
        )
