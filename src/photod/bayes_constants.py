from pydantic import BaseModel


class BayesConstants(BaseModel):
    FeHmin: float = -2.5  # the range of considered [Fe/H], in priors and elsewhere
    FeHmax: float = 1.0
    FeHNpts: int = 36  # defines the grid step for [Fe/H]
    MrFaint: float = 17.0  # the range of considered Mr, in priors and elsewhere
    MrBright: float = -2.0
    MrNpts: int = 96  # defines the grid step for Mr
    rmagBinWidth: float = 0.5  # could be dependent on rmag and larger for bright mags...
    # parameters for pre-computed TRILEGAL-based prior maps
    rmagMin: int = 14
    rmagMax: int = 27
    rmagNsteps: int = 27
    # parameters for dust extinction grid when fitting 3D (Mr, FeH, Ar)
    # prior for Ar is flat from (0*Ar+0.0) to (1.3*Ar+0.1) with a step of 0.01 mag ***
    # where 1.3 <= ArCoef0, 0.1 <= ArCoef1 and 0.01 mag <= ArCoef2 are set here (or user supplied)
    # ArMin: ArCoeff3*Ar + ArCoeff4
    ArCoeff0: float = 1.3
    ArCoeff1: float = 0.1
    ArCoeff2: float = 0.01
    # allow Ar as small as 0
    ArCoeff3: float = 0.0
    ArCoeff4: float = 0.0
