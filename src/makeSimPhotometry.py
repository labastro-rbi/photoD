import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import LocusTools as lt 
import BayesTools as bt 
import PlotTools as pt
from astropy.table import Table, vstack


def makeSimLSSTcatFromTRILEGAL(infile, MSRGlocus, WDlocusH, WDlocusHe, outfile, verbose=False):

    ## 1) given rmag, Ar and DM from TRILEGAL, generate Mr 
    ## 2) with this Mr and FeH, generate colors from locus data  
    ## 3) with original r magnitude, generate other magnitudes from colors
    ## 4) given magnitudes, generate errors for LSST
    ## 5) generate observed mags
    ## 6) regenerate "observed" errors from "observed" magnitudes
    ## 7) and return data frame 

    ## 1)
    ## read catalog produced at Astro Lab, using dumpCatalogFromPatch in makePriors.py
    ## magnitudes (umag, gmag...) are not corrected for extinction, but colors are
    cat = lt.readAstroLabCatalog(infile, verbose=verbose)

    # save original TRILEGAL magnitudes
    for b in ['u', 'g', 'r', 'i', 'z']:  
        cat[b+'magTL'] = cat[b+'mag']
        ## and this is a hack to "fix" TRILEGAL [Fe/H] distribution to be more similar to SDSS measurements
    if (1):
        # shift thin disk stars to lower [Fe/H] by 0.3 dex
        cat['FeH'] = np.where(cat['GC']==1, cat['FeH']-0.3, cat['FeH'])
        # shift halo stars to higher [Fe/H] by 0.1 dex
        cat['FeH'] = np.where(cat['GC']==3, cat['FeH']+0.1, cat['FeH'])

    ## separate MS/RGs from white dwarfs 
    catMSRG = cat[cat['pop']<9]
    catWD = cat[cat['pop']==9]
    if verbose: print('MSRG:', np.size(catMSRG), 'WD:', np.size(catWD))

    ## 2)     
    ## locus data
    L10 = lt.readSDSSDSEDlocus(MSRGlocus, fixForStripe82=False)
    # white dwarfs
    HWD = lt.readWDlocus(WDlocusH, verbose=verbose)
    HeWD = lt.readWDlocus(WDlocusHe, verbose=verbose)    

    # limit FeH and Mr to values supported by LSSTlocus (DSED version)
    FeHmin = -2.50
    FeHmax = 0.50
    MrRGcut = -2.0  # limit from TRILEGAL priors 
    MrMax = 16.0    # limit from SDSS locus 
    # apply FeH and Mr cuts above: 
    tGoodMSRG = catMSRG[(catMSRG['FeH']>=FeHmin)&(catMSRG['FeH']<=FeHmax)&(catMSRG['Mr']>MrRGcut)&(catMSRG['Mr']<MrMax)]
    if verbose: print('downselected to:', np.size(tGoodMSRG), 'fraction:', np.size(tGoodMSRG)/np.size(catMSRG)) 

    ## assign colors to MSRG subsample
    if verbose: print('Generating colors for MS/RG stars: the longest step...')
    lt.getColorsFromMrFeHDSED(L10, tGoodMSRG)
    if verbose: print('Finished color assignment for MS/RG stars')
    ## assign colors to WD subsample
    fHe = 0.2    ## 0.1 in LSST Science Book, section 6.11.6 but we need to emphasize He more
    lt.getWDcolorsFromMr(HWD, HeWD, fHe, catWD)    
    ## and now can merge MSRG and WD catalogs 
    tGoodOK =  vstack([tGoodMSRG, catWD])
    if verbose: print('Generated model colors, Nsource=', len(tGoodOK))

    ## 3)
    # let's rename original assigned colors (and add suffix 0 since they don't have dust reddening) 
    tGoodOK['ugSL0'] = tGoodOK['ug']
    tGoodOK['grSL0'] = tGoodOK['gr']
    tGoodOK['riSL0'] = tGoodOK['ri']
    tGoodOK['izSL0'] = tGoodOK['iz'] 
    tGoodOK['giSL0'] = tGoodOK['grSL0'] + tGoodOK['riSL0']
        
    ## tGoodOK['ug0'] etc. are dust-free colors, now add reddening using Ar along the sightline from TRILEGAL 
    ## and standard extinction coefficients (from Berry+2012)
    # first setup of dust extinction in each band
    C = lt.extcoeff()
    for b in ['u', 'g', 'i', 'z']:  
        tGoodOK['A'+b] = C[b]*tGoodOK['Ar']  
    # and now redden input colors (generated from stellar locus, they are still without noise) 
    tGoodOK['ugSL'] = tGoodOK['ugSL0'] + tGoodOK['Au'] - tGoodOK['Ag'] 
    tGoodOK['grSL'] = tGoodOK['grSL0'] + tGoodOK['Ag'] - tGoodOK['Ar'] 
    tGoodOK['riSL'] = tGoodOK['riSL0'] + tGoodOK['Ar'] - tGoodOK['Ai'] 
    tGoodOK['izSL'] = tGoodOK['izSL0'] + tGoodOK['Ai'] - tGoodOK['Az'] 

    ## and define here the final sample     
    # LSST-motivated faint limit for the output catalog (removes faint stars, ~28%)
    rmagMax = 26.0 
    giMin = -1.0
    giMax = 5.0
    tGood4sims = tGoodOK[(tGoodOK['giSL0']>giMin)&(tGoodOK['giSL0']<giMax)&(tGoodOK['ugSL0']>-1)&(tGoodOK['rmagTL']<rmagMax)]
    if verbose: print('Final sample:', np.size(tGood4sims), np.size(tGood4sims)/np.size(tGoodOK))

    ## 3)
    ## simulated (noise-free) magnitudes, anchored to TRILEGAL's original r band magnitude: tGood4sims['rmag']
    ## note that colors 0 (e.g. tGood4sims['gr']) are DUST-FREE COLORS generated in lt.getColorsFromMrFeH,
    ## while colors/magnitudes without "0" include dust reddening/extinction 
    ## tGood4sims['rmag'] comes directly from TRILEGAL (and thus it includes dust extinction)
    ## generate other magnitudes (ISM extinction included via rmag from TRILEGAL) 
    tGood4sims['rmagSL'] = tGood4sims['rmagTL']
    tGood4sims['gmagSL'] = tGood4sims['rmagSL'] + tGood4sims['grSL']
    tGood4sims['umagSL'] = tGood4sims['gmagSL'] + tGood4sims['ugSL']
    tGood4sims['imagSL'] = tGood4sims['rmagSL'] - tGood4sims['riSL']
    tGood4sims['zmagSL'] = tGood4sims['imagSL'] - tGood4sims['izSL']

    ## 4)
    ## generate errors expected for LSST coadded depth (per Bianco+2022 paper)
    ##   note: errors are generated using true dust-extincted magnitudes generated in 3
    errorsTrue = lt.getLSSTm5(tGood4sims, depth='coadd', magVersion=True, suffix='SL')

    ## 5)
    ## generate observed mags (by drawing random gaussian noise with mag-dependent std)
    ObsMag = {}
    for b in ['u', 'g', 'r', 'i', 'z']:  
        tGood4sims[b+'magErrSL'] = errorsTrue[b]
        noise = np.random.normal(0, tGood4sims[b+'magErrSL'])   
        ### this is a hack to prevent super faint stars ending up very bright ###
        minErr = np.where(np.abs(noise)>1, 1, noise)
        # adding noise to true noise-free dust-extincted magnitudes
        tGood4sims[b+'magObs'] = tGood4sims[b+'magSL'] + minErr
        ObsMag[b] = tGood4sims[b+'magObs']

    # errors for "SL" colors
    tGood4sims['ugErrSL'] = np.sqrt(tGood4sims['umagErrSL']**2 + tGood4sims['gmagErrSL']**2)
    tGood4sims['grErrSL'] = np.sqrt(tGood4sims['gmagErrSL']**2 + tGood4sims['rmagErrSL']**2)
    tGood4sims['riErrSL'] = np.sqrt(tGood4sims['rmagErrSL']**2 + tGood4sims['imagErrSL']**2)
    tGood4sims['izErrSL'] = np.sqrt(tGood4sims['imagErrSL']**2 + tGood4sims['zmagErrSL']**2)
        
    # now get errors derived from "observed" magnitudes
    # IMPORTANT: errors are NOT generated using original noise-free magnitudes
    ObsMagErr = lt.getLSSTm5err(ObsMag)
    for b in ['u', 'g', 'r', 'i', 'z']:  
        tGood4sims[b+'magErrObs'] = ObsMagErr[b]  

    # generate observed colors (with and without dust extinction)
    tGood4sims['ugObs'] = tGood4sims['umagObs'] - tGood4sims['gmagObs'] 
    tGood4sims['grObs'] = tGood4sims['gmagObs'] - tGood4sims['rmagObs'] 
    tGood4sims['riObs'] = tGood4sims['rmagObs'] - tGood4sims['imagObs'] 
    tGood4sims['izObs'] = tGood4sims['imagObs'] - tGood4sims['zmagObs'] 
    tGood4sims['giObs'] = tGood4sims['grObs'] + tGood4sims['riObs'] 

    # correct colors for ISM dust reddening
    tGood4sims['ugObs0'] = tGood4sims['ugObs'] - (tGood4sims['Au']-tGood4sims['Ag'])
    tGood4sims['grObs0'] = tGood4sims['grObs'] - (tGood4sims['Ag']-tGood4sims['Ar'])
    tGood4sims['riObs0'] = tGood4sims['riObs'] - (tGood4sims['Ar']-tGood4sims['Ai'])
    tGood4sims['izObs0'] = tGood4sims['izObs'] - (tGood4sims['Ai']-tGood4sims['Az'])
    tGood4sims['giObs0'] = tGood4sims['giObs'] - (tGood4sims['Ag']-tGood4sims['Ai'])

    # errors for observed colors 
    tGood4sims['ugErrObs'] = np.sqrt(tGood4sims['umagErrObs']**2 + tGood4sims['gmagErrObs']**2) 
    tGood4sims['grErrObs'] = np.sqrt(tGood4sims['gmagErrObs']**2 + tGood4sims['rmagErrObs']**2) 
    tGood4sims['riErrObs'] = np.sqrt(tGood4sims['rmagErrObs']**2 + tGood4sims['imagErrObs']**2) 
    tGood4sims['izErrObs'] = np.sqrt(tGood4sims['imagErrObs']**2 + tGood4sims['zmagErrObs']**2)

    # done... 
    if outfile=='':
        if verbose: print('done!') 
        return tGood4sims
    else:
        ## store in a file 
        if verbose: print('storing to file:', outfile)
        writeSimCatalog(tGood4sims, outfile)


def writeSimCatalog(tGood4sims, outfile):
    fout = open(outfile, "w")
    fout.write("      glon       glat      comp   logg    FeH      Mr      DM      Ar ")
    fout.write(" rmagObs0   ug0     gr0     ri0     iz0    rmag   ugObs   grObs   riObs   izObs")
    fout.write("   uErr   gErr   rErr   iErr   zErr    ugSL    grSL    riSL    izSL")
    fout.write(" ugErrSL grErrSL riErrSL izErrSL \n")
    for i in range(0,np.size(tGood4sims)):
        # input values from TRILEGAL
        r1 = tGood4sims['glon'][i]
        r2 = tGood4sims['glat'][i]
        r3 = tGood4sims['GC'][i]
        r4 = tGood4sims['logg'][i]
        r5 = tGood4sims['FeH'][i]
        r6 = tGood4sims['Mr'][i]
        r7 = tGood4sims['DM'][i]
        r8 = tGood4sims['Ar'][i]  
        s = str("%12.8f " % r1) + str("%12.8f  " % r2) + str("%3.0f  " % r3) + str("%6.2f  " % r4)
        s = s + str("%5.2f  " % r5) + str("%6.2f  " % r6) + str("%6.2f  " % r7) + str("%6.3f  " % r8)  
        # observed colors, corrected for ISM extinction, same errors as for ugObs... below
        r1 = tGood4sims['rmagTL'][i]
        r2 = tGood4sims['ugObs0'][i] 
        r3 = tGood4sims['grObs0'][i] 
        r4 = tGood4sims['riObs0'][i] 
        r5 = tGood4sims['izObs0'][i] 
        s = s + str("%6.2f" % r1) + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5)
        # observed colors, with ISM extinction included
        r1 = tGood4sims['rmagObs'][i]
        r2 = tGood4sims['ugObs'][i] 
        r3 = tGood4sims['grObs'][i] 
        r4 = tGood4sims['riObs'][i] 
        r5 = tGood4sims['izObs'][i] 
        s = s + str("%8.2f" % r1) + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5)
        # errors for observed mags, generated from observed mags
        r1 = tGood4sims['umagErrObs'][i]
        r2 = tGood4sims['gmagErrObs'][i]
        r3 = tGood4sims['rmagErrObs'][i]
        r4 = tGood4sims['imagErrObs'][i]
        r5 = tGood4sims['zmagErrObs'][i]
        s = s + str("%7.3f" % r1) + str("%7.3f" % r2) + str("%7.3f" % r3) + str("%7.3f" % r4) + str("%7.3f" % r5)
        # true input values (ISM extinction included)
        r2 = tGood4sims['ugSL'][i] 
        r3 = tGood4sims['grSL'][i] 
        r4 = tGood4sims['riSL'][i] 
        r5 = tGood4sims['izSL'][i] 
        s = s + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5)
        # true errors based on input mag values above
        r2 = tGood4sims['ugErrSL'][i] 
        r3 = tGood4sims['grErrSL'][i] 
        r4 = tGood4sims['riErrSL'][i] 
        r5 = tGood4sims['izErrSL'][i] 
        s = s + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5)
        s = s + "\n"
        fout.write(s)             
    fout.close() 
