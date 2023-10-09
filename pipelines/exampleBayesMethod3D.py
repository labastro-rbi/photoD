#!/usr/bin/env python
# coding: utf-8

import sys
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import hstack
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
from scipy import optimize
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
 
# importing plotting and locus tools: 
sys.path.append('../src')
import LocusTools as lt
import BayesTools as bt
import PlotTools as pt


timeStart = get_ipython().getoutput('date +%s  ')   

### read photometric catalog with stars
## read TRILEGAL sim-based data file, augmented with LSST colors 
## the input data are limited to 0.2 < g-i < 3.0 and
## -2.5 < FeH < 0 and Mr > -1.0 and log(g) < 7 and rmag < 26 
sims = lt.readTRILEGALLSST(inTLfile='../data/simCatalog_three_pix_triout_chiTest4.txt', chiTest=True)

### read stellar locus parametrization
LSSTlocus = lt.LSSTsimsLocus(fixForStripe82=False)
## select the color range where we expect main sequence and red giants
OKlocus = LSSTlocus[(LSSTlocus['gi']>0.2)&(LSSTlocus['gi']<3.55)]   # gives MrMax < 15 (rerun priors!)

### subsample Mr and FeH grids (linear speed-up but somewhat lower accuracy)
kMr = 10
kFeH = 2 
locusData = lt.subsampleLocusData(OKlocus, kMr, kFeH)

### process data...
catalog = sims
fitColors = ('ug', 'gr', 'ri', 'iz')  
priorsRootName = '../data/TRILEGALpriors/priors'
outfile = '../data/simCatalog_three_pix_triout_chiTest4_BayesEstimates3Dsparse10_2_small.txt'

### make 3D locus list with three ArGrid limits and resolutions
ArGridList, locus3DList = lt.get3DmodelList(locusData, fitColors)

## all set, now select range of stars to process 
iStart = 0
iEnd = 1000    # if <0: do all stars
myStars = [0, 10, 100, 1000, 10000]  # for method illustration, make plots for these stars 
myStars = [] # no plots 
verb=False 

timeMid = get_ipython().getoutput('date +%s  ')    
runTime = int(timeMid[0])-int(timeStart[0])
print('Setup in', runTime,'seconds.')

### and call workhorse... ### 
bt.makeBayesEstimates3D(catalog, fitColors, locusData, locus3DList, ArGridList, priorsRootName, outfile, iStart, iEnd, myStars, verbose=verb)
  
timeEnd = get_ipython().getoutput('date +%s  ')    
runTime = int(timeEnd[0])-int(timeMid[0])
print('Processed stars in', runTime,'seconds.')
