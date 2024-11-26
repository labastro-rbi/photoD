from pydantic import BaseModel


class BayesConstants(BaseModel):
    FeHmin: float  # the range of considered [Fe/H], in priors and elsewhere
    FeHmax: float
    FeHNpts: int  # defines the grid step for [Fe/H]
    MrFaint: float  # the range of considered Mr, in priors and elsewhere
    MrBright: float
    MrNpts: int  # defines the grid step for Mr
    rmagBinWidth: float  # could be dependent on rmag and larger for bright mags...
    # parameters for pre-computed TRILEGAL-based prior maps
    rmagMin: int
    rmagMax: int
    rmagNsteps: int
    # parameters for dust extinction grid when fitting 3D (Mr, FeH, Ar)
    # prior for Ar is flat from (0*Ar+0.0) to (1.3*Ar+0.1) with a step of 0.01 mag ***
    # where 1.3 <= ArCoef0, 0.1 <= ArCoef1 and 0.01 mag <= ArCoef2 are set here (or user supplied)
    # ArMin: ArCoeff3*Ar + ArCoeff4
    ArCoeff0: float
    ArCoeff1: float
    ArCoeff2: float
    # allow Ar as small as 0
    ArCoeff3: float
    ArCoeff4: float

    def get_arcoeff_dict(self) -> dict[int, float]:
        return {
            0: self.ArCoeff0,
            1: self.ArCoeff1,
            2: self.ArCoeff2,
            3: self.ArCoeff3,
            4: self.ArCoeff4,
        }
