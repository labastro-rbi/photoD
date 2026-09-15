import numpy as np
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import BayesTools as bt
import PlotTools as pt

# def MSlocus(gi, FeH):
# def RGlocus(gi, FeH): 
# def GClocusGiants(Mr, FeH):
# def CoveyMSlocus(): 
# def readTRILEGAL():
# def readTRILEGALLSST():
# def readTRILEGALLSSTestimates():
# def readKarloMLestimates3(inKfile, simtype):
# def LSSTsimsLocus(fixForStripe82=True): 
# def BHBlocus(): 
# def photoFeH(ug, gr): 
# def getMr(gi, FeH):
# def extcoeff(): 
# def getColorsFromMrFeH(L, Lvalues):
# def getColorsFromGivenMrFeH(myMr, myFeH, L, gp, colors=""):
# def getLocusChi2colors(colorNames, Mcolors, Ocolors, Oerrors):
# def logLfromObsColors(Mr, FeH, ObsColors, ObsColErrors, gridProperties, SDSScolors):
# def getFitResults(i, gridStride, Ntest, colors, locusGrid, ObsColorErr, locusFitMS, locusFitRG, results):
# def extractBestFit(chi2vals, locusGrid):
# def fitPhotoD(i, gridStride, colors, data2fit, locusFitMS, locusFitRG, results):
# def getPhotoDchi2map(i, colors, data2fit, locus):
# def getPhotoDchi2map3D(i, colors, colorReddCoeffs, data2fit, locus, ArCoeff, masterLocus=False):
# def get3DmodelList(locusData, fitColors):
# def make3DlocusFast(locus3D0, ArGrid, colors, colorCorrection, FeH1d, Mr1d):
# def make3DlocusList(locusData, fitColors, ArGridList):
# def make3Dlocus(locus, ArGrid, colors, colCorr):
# def fitMedians(x, y, xMin, xMax, Nbin, verbose=1): 
# def basicStats(df, colName):
# def getMedianSigG(basicStatsLine):
# def makeStatsTable0(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):
# def makeStatsTable(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):
# def getLSSTm5(data, depth='coadd', magVersion=False, suffix=''):
# def getLSSTm5err(mags, depth='coadd'):
  
def MSlocus(gi, FeH):
        ## main sequence SDSS color locus, as a function of [Fe/H], from
        ## Yuan et al (2015, ApJ 799, 134): "Stellar loci I. Metalicity dependence and intrinsic widths" 
        ## https://iopscience.iop.org/article/10.1088/0004-637X/799/2/134/pdf
        ## 
        ## f(x,y) = a0 + a1y+ a2y2 + a3y3 + a4x + a5xy + a6xy2 + a7x2 + a8yx2 + a9x3,
        ##     where x≡g−i, y≡[Fe/H], and f = g-r, r-i, i-z (see below for u-g)
        ## 
        ## for u-g color, there are five more terms (and a different expression) 
        ## f(x,y) =  b0    + b1y    + b2y2    + b3y3  + b4y4 +
        ##           b5x   + b6yx   + b7y2x   + b8y3x +
        ##           b9x2  + b10yx2 + b11y2x2 +
        ##           b12x3 + b13yx3 +
        ##           b14x4
        ## 
        ##  valid for 0.3 < g-i < 1.6 and −2.0 < [Fe/H] < 0.0  
        ## 
        ## populate the arrays of coefficients (Table 1 in Yuan et al) 
        # u−g
        b = {}  # renamed from a here for clarity wrt other three colors below 
        b[0] =  1.5003
        b[1] =  0.0011
        b[2] =  0.1741
        b[3] =  0.0910
        b[4] =  0.0181
        b[5] = -3.2190
        b[6] =  1.1675
        b[7] =  0.0875
        b[8] =  0.0031
        b[9] =  7.9725
        b[10] = -0.8190
        b[11] = -0.0439
        b[12] = -5.2923
        b[13] =  0.1339
        b[14] =  1.1459
        # g-r
        a = {}
        a[1,0] =  0.0596
        a[1,1] =  0.0348
        a[1,2] =  0.0239
        a[1,3] =  0.0044
        a[1,4] =  0.6070
        a[1,5] =  0.0261
        a[1,6] = -0.0044
        a[1,7] =  0.1532
        a[1,8] = -0.0136
        a[1,9] = -0.0613
        # r-i
        a[2,0] = -0.0596
        a[2,1] = -0.0348
        a[2,2] = -0.0239
        a[2,3] = -0.0044
        a[2,4] =  0.3930
        a[2,5] = -0.0261
        a[2,6] =  0.0044
        a[2,7] = -0.1532
        a[2,8] =  0.0136
        a[2,9] =  0.0613        
        # i-z
        a[3,0] = -0.1060
        a[3,1] = -0.0357
        a[3,2] = -0.0123
        a[3,3] = -0.0017
        a[3,4] =  0.2543
        a[3,5] = -0.0010
        a[3,6] = -0.0050
        a[3,7] = -0.0381
        a[3,8] = -0.0071
        a[3,9] =  0.0030

        ## evaluate colors
        SDSScolors = ('ug', 'gr', 'ri', 'iz')
        color = {}
        for c in SDSScolors:
            color[c] = 0*(gi+FeH)
            j = SDSScolors.index(c)
            x = gi
            y = FeH
            if (c=='ug'):
                color[c] += b[0] + b[1]*y + b[2]*y**2 + b[3]*y**3 + b[4]*y**4 
                color[c] += b[5]*x + b[6]*x*y + b[7]*x*y**2 + b[8]*x*y**3 + b[9]*x**2
                color[c] += b[10]*x**2*y + b[11]*x**2*y**2 + b[12]*x**3 + b[13]*x**3*y + b[14]*x**4
            else:
                color[c] += a[j,0] + a[j,1]*y + a[j,2]*y**2 + a[j,3]*y**3 + a[j,4]*x + a[j,5]*x*y
                color[c] += a[j,6]*x*y**2 + a[j,7]*x**2 + a[j,8]*y*x**2 + a[j,9]*x**3
        color['gi'] = gi
        color['FeH'] = 0*gi + FeH       
        return color


def RGlocus(gi, FeH): 
        ## red giant SDSS color locus, as a function of [Fe/H], from
        ## Zhang et al (2021, RAA, 21, 12, 319): "Stellar loci IV. red giant stars" 
        ## https://iopscience.iop.org/article/10.1088/1674-4527/21/12/319/pdf
        ## 
        ## f(x,y) = a0 + a1y+ a2y2 + a3y3 + a4x + a5xy + a6xy2 + a7x2 + a8yx2 + a9x3,
        ##     where x≡g−i, y≡[Fe/H], and f = ug, gr, ri, iz
        ##  valid for 0.55 < g-i < 1.2 and −2.5 < [Fe/H] < -0.3 
        ## 
        ## populate the array of coefficients (Table 1 in Zhang et al) 
        a = {}
        # u−g
        a[0,0] =  1.4630
        a[0,1] =  0.3132
        a[0,2] = -0.0105
        a[0,3] = -0.0224
        a[0,4] = -1.5851
        a[0,5] = -0.2423
        a[0,6] = -0.0372
        a[0,7] =  2.8655
        a[0,8] =  0.0958
        a[0,9] = -0.7469
        # g-r
        a[1,0] =  0.0957
        a[1,1] =  0.0370
        a[1,2] =  0.0120
        a[1,3] =  0.0020
        a[1,4] =  0.5272
        a[1,5] = -0.0026
        a[1,6] =  0.0019
        a[1,7] =  0.1645
        a[1,8] =  0.0057
        a[1,9] = -0.0488        
        # r-i
        a[2,0] = -0.0957
        a[2,1] = -0.0370
        a[2,2] = -0.0120
        a[2,3] = -0.0020
        a[2,4] =  0.4728
        a[2,5] =  0.0026
        a[2,6] = -0.0019
        a[2,7] = -0.1645
        a[2,8] = -0.0057
        a[2,9] =  0.0488        
        # i-z
        a[3,0] = -0.0551
        a[3,1] = -0.0229
        a[3,2] = -0.0165
        a[3,3] = -0.0033
        a[3,4] =  0.0762
        a[3,5] = -0.0365
        a[3,6] = -0.0006
        a[3,7] =  0.1899
        a[3,8] =  0.0244
        a[3,9] = -0.0805
        ## evaluate colors
        SDSScolors = ('ug', 'gr', 'ri', 'iz')
        color = {}
        for c in SDSScolors:
            color[c] = 0*(gi+FeH)
            j = SDSScolors.index(c)
            x = gi
            y = FeH
            color[c] += a[j,0] + a[j,1]*y + a[j,2]*y**2 + a[j,3]*y**3 + a[j,4]*x + a[j,5]*x*y
            color[c] += a[j,6]*x*y**2 + a[j,7]*x**2 + a[j,8]*y*x**2 + a[j,9]*x**3
        color['gi'] = gi
        color['FeH'] = 0*gi + FeH       
        return color


def GClocusGiants(Mr, FeH):
    ## fits to the RGB branch for globular clusters observed
    ## by SDSS: M2, M3, M5, M13, M15, and M53
    ## the median RGB color is determined for each cluster 
    ## using two Mr bins: 1.5 < Mr < 2 and -0.5 < Mr < 0
    ## for each bin, color vs FeH dependence is fit by a 
    ## linear function
    ## the results are roughly valid for 2.5 < Mr < -1 and
    ## are accurate to a few hundreths of a mag (better 
    ## accuracy for redder colors and brighther bin 
    ## fits as functions of Mr and FeH 
    ##  color = A + B * FeH + C * Mr + D * FeH * Mr
    color = {}
    color['ug'] =  1.9805 +0.2865*FeH -0.2225*Mr -0.0425*FeH*Mr
    color['gr'] =  0.7455 +0.0835*FeH -0.1225*Mr -0.0325*FeH*Mr
    color['ri'] =  0.3100 +0.0280*FeH -0.0500*Mr -0.0100*FeH*Mr
    color['iz'] =  0.0960 +0.0000*FeH -0.0200*Mr +0.0000*FeH*Mr
    color['Mr'] = 0*color['iz'] + Mr
    color['FeH'] = 0*color['iz'] + FeH
    return color

def CoveyMSlocus(): 
        ## STELLAR LOCUS IN THE SDSS-2MASS PHOTOMETRIC SYSTEM
        ## read and return colors from
        ## http://faculty.washington.edu/ivezic/sdss/catalogs/tomoIV/Covey_Berry.txt
        ## for more details see the header (note that 2MASS is on Vega system!)
        ## according to Yuan et al (2015, ApJ 799, 134), this locus approximately corresponds to FeH = -0.5 
        colnames = ['Nbin', 'gi', 'Ns', 'ug', 'Eug', 'gr', 'Egr', 'ri', 'Eri', 'iz', 'Eiz', \
                     'zJ', 'EzJ', 'JH', 'EJH', 'HK', 'EHK']
        coveyMS = Table.read('./Covey_Berry.txt', format='ascii', names=colnames)
        return coveyMS 


def readTRILEGAL(infile=''):
        if (infile == ''):
                # Dani Chao's example file produced using TRILEGAL simulation
                infile = '../data/three_pix_triout.dat'
                print('reading from default file:', infile)
        else:
                print('reading from:', infile)
        # Dani: gall galb Gc logAge M_H mu0 Av logg gmag rmag imag umag zmag
        colnames = ['glon', 'glat', 'comp', 'logage', 'FeH', 'DM', 'Av', 'logg', 'gmag', 'rmag', 'imag', 'umag', 'zmag']
        # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
        # logage with age in years
        # DM = m-M is called true distance modulus in DalTio+(2022), so presumably extinction is not included
        # and thus Mr = rmag - Ar - DM - 5  
        ## read TRILEGAL simulation (per healpix, as extracted by Dani, ~1-2M stars)
        trilegal = Table.read(infile, format='ascii', names=colnames)
        # dust extinction: Berry+ give Ar = 2.75E(B-V) and DalTio+ used Av=3.10E(B-V)
        trilegal['Ar'] = 2.75 * trilegal['Av'] / 3.10   
        C = extcoeff()
        # correcting colors for extinction effects 
        trilegal['ug'] = trilegal['umag'] - trilegal['gmag'] - (C['u'] - C['g'])*trilegal['Ar']  
        trilegal['gr'] = trilegal['gmag'] - trilegal['rmag'] - (C['g'] - C['r'])*trilegal['Ar']   
        trilegal['ri'] = trilegal['rmag'] - trilegal['imag'] - (C['r'] - C['i'])*trilegal['Ar']   
        trilegal['iz'] = trilegal['imag'] - trilegal['zmag'] - (C['i'] - C['z'])*trilegal['Ar']   
        trilegal['gi'] = trilegal['gr'] + trilegal['ri']     
        return trilegal 


def readTRILEGALLSST(inTLfile='default', chiTest=False):
        ## read TRILEGAL simulation augmented with LSST colors, see TRILEGAL_makeTestFile.ipynb 
        colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
        colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr']
        if chiTest:
                # these are stellar locus colors and errors generated from resulting noise-free magnitudes
                colnames = colnames + ['ugSL', 'grSL', 'riSL', 'izSL']
                colnames = colnames + ['ugErrSL', 'grErrSL', 'riErrSL', 'izErrSL']
        # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
        # Mr = rmag - Ar - DM - 5
        # rmag0, ug0...iz0: observed values without dust extinction (but include photometric noise, except for rmag0)
        # rmag, ug...iz: dust extinction included
        # uErr...zErr: photometric noise
        if (inTLfile=='default'):
            print('DEFAULT OK?')
            sims = Table.read('../data/TRILEGAL_three_pix_triout_V1.txt', format='ascii', names=colnames)
        else:
            print('READING FROM', inTLfile)
            sims = Table.read(inTLfile, format='ascii', names=colnames)
        sims['gi0'] = sims['gr0'] + sims['ri0']  
        sims['gi'] = sims['gr'] + sims['ri']
        print(np.size(sims), 'read from', inTLfile)
        # these are (noisy) dust-reddened colors and thus dust-extincted magnitudes
        sims['gmag'] = sims['rmag'] + sims['gr']  
        sims['umag'] = sims['gmag'] + sims['ug']
        sims['imag'] = sims['rmag'] - sims['ri']
        sims['zmag'] = sims['imag'] - sims['iz']
        # color errors
        sims['ugErr'] = np.sqrt(sims['uErr']**2 + sims['gErr']**2)
        sims['grErr'] = np.sqrt(sims['gErr']**2 + sims['rErr']**2)
        sims['riErr'] = np.sqrt(sims['rErr']**2 + sims['iErr']**2)
        sims['izErr'] = np.sqrt(sims['iErr']**2 + sims['zErr']**2)
        return sims


def readTRILEGALLSSTestimates(infile='../data//TRILEGAL_three_pix_triout_BayesEstimates.txt', b3D=False):
        ## read FeH and Mr estimates and their uncertainties 
        colnames = ['glon', 'glat', 'FeHEst', 'FeHUnc', 'MrEst', 'MrUnc', 'chi2min', 'MrdS', 'FeHdS']  
        if b3D:
            colnames = ['glon', 'glat', 'FeHEst', 'FeHUnc', 'MrEst', 'MrUnc', 'ArEst', 'ArUnc', 'chi2min', 'MrdS', 'FeHdS', 'ArdS', 'counter']
        simsEst = Table.read(infile, format='ascii', names=colnames)
        print(np.size(simsEst), 'read from', infile)
        return simsEst


def readKarloMLestimates(multiMethod=True, inKfile='default'):
        if (inKfile=='default'):
            print('DEFAULT OK?')
            file = 'ML_predictions_TRILEGAL_three_pix_triout_V1.txt'
            ## file includes:
            colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
            colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr', 'gi0', 'gi']
            colnames = colnames + ['test_set', 'naive_single_Mr', 'naive_single_Ar', 'naive_single_FeH']
            colnames = colnames + ['naive_multi_Mr', 'naive_multi_Ar', 'naive_multi_FeH']
            colnames = colnames + ['sampling_single_Mr', 'sampling_single_MrErr', 'sampling_single_Ar']
            colnames = colnames + ['sampling_single_ArErr', 'sampling_single_FeH', 'sampling_single_FeHErr']
            colnames = colnames + ['sampling_multi_Mr', 'sampling_multi_MrErr', 'sampling_multi_Ar']
            colnames = colnames + ['sampling_multi_ArErr', 'sampling_multi_FeH', 'sampling_multi_FeHErr']
        else:
            print('READING FROM', inKfile)
            file = inKfile
            ## file includes:
            colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
            colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr', 'gi0', 'gi', 'test_set']
            colnames = colnames + ['naive_single_Mr', 'naive_single_MrErr', 'naive_single_Ar', 'naive_single_ArErr', 'naive_single_FeH', 'naive_single_FeHErr']
            colnames = colnames + ['naive_multi_Mr', 'naive_multi_MrErr', 'naive_multi_Ar', 'naive_multi_ArErr', 'naive_multi_FeH', 'naive_multi_FeHErr']
            colnames = colnames + ['simple_single_Mr', 'simple_single_MrErr', 'simple_single_Ar', 'simple_single_ArErr', 'simple_single_FeH', 'simple_single_FeHErr']
            colnames = colnames + ['single_multi_Mr', 'single_multi_MrErr', 'single_multi_Ar', 'single_multi_ArErr', 'single_multi_FeH', 'single_multi_FeHErr']
            colnames = colnames + ['sampling_single_Mr', 'sampling_single_MrErr', 'sampling_single_Ar', 'sampling_single_ArErr', 'sampling_single_FeH', 'sampling_single_FeHErr']
            colnames = colnames + ['sampling_multi_Mr', 'sampling_multi_MrErr', 'sampling_multi_Ar', 'sampling_multi_ArErr', 'sampling_multi_FeH', 'sampling_multi_FeHErr']
            ## yet another new format..
            colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
            colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr', 'gi0', 'gi', 'test_set']
            colnames = colnames + ['simple_single_Mr', 'simple_single_MrErr', 'simple_single_Ar', 'simple_single_ArErr', 'simple_single_FeH', 'simple_single_FeHErr']
           
        ## read FeH and Mr estimates and fake their uncertainties 
        simsML = Table.read(file, format='ascii', names=colnames)
        if (inKfile=='default'):
           simsML['MrUnc'] = -1
           simsML['FeHUnc'] = -1
           # single vs. multi
           if (multiMethod):
                simsML['MrEstML'] = simsML['naive_multi_Mr']
                simsML['FeHEstML'] = simsML['naive_multi_FeH']
           else:
                simsML['MrEstML'] = simsML['naive_single_Mr']
                simsML['FeHEstML'] = simsML['naive_single_FeH']
                simsML['MrUnc'] = simsML['naive_single_MrErr']
                simsML['FeHUnc'] = simsML['naive_single_FeHErr'] 
        else:
             print('using simple_single with ML uncertainties')
             simsML['MrEstML'] = simsML['simple_single_Mr']
             simsML['MrUnc'] = simsML['simple_single_MrErr']
             simsML['FeHEstML'] = simsML['simple_single_FeH']
             simsML['FeHUnc'] = simsML['simple_single_FeHErr']
             
        print(np.size(simsML), 'read from', file)
        return simsML



def readKarloMLestimates3(inKfile, simtype):
        print('READING FROM', inKfile)
        file = inKfile       
        ## file includes:
        colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
        colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr', 'gi0', 'gi', 'test_set']
        colnames = colnames + ['Mr_a', 'MrErr_a', 'Ar_a', 'ArErr_a', 'FeH_a', 'FeHErr_a']
        if simtype=='a':
                print('assuming truncated format from Sep 23')
        else: 
                colnames = colnames + ['Mr_b', 'MrErr_b', 'Ar_b', 'ArErr_b', 'FeH_b', 'FeHErr_b']
                colnames = colnames + ['Mr_c', 'MrErr_c', 'Ar_c', 'ArErr_c', 'FeH_c', 'FeHErr_c']          
        ## read FeH and Mr estimates and fake their uncertainties 
        simsML = Table.read(file, format='ascii', names=colnames)
        if simtype=='a':
                simsML['MrEstML'] = simsML['Mr_a']
                simsML['MrUnc'] = simsML['MrErr_a']
                simsML['FeHEstML'] = simsML['FeH_a']
                simsML['FeHUnc'] = simsML['FeHErr_a']
                simsML['ArEstML'] = simsML['Ar_a']
                simsML['ArUnc'] = simsML['ArErr_a']
        if simtype=='b':
                simsML['MrEstML'] = simsML['Mr_b']
                simsML['MrUnc'] = simsML['MrErr_b']
                simsML['FeHEstML'] = simsML['FeH_b']
                simsML['FeHUnc'] = simsML['FeHErr_b']
                simsML['ArEstML'] = simsML['Ar_b']
                simsML['ArUnc'] = simsML['ArErr_b']
        if simtype=='c':
                simsML['MrEstML'] = simsML['Mr_c']
                simsML['MrUnc'] = simsML['MrErr_c']
                simsML['FeHEstML'] = simsML['FeH_c']
                simsML['FeHUnc'] = simsML['FeHErr_c']
                simsML['ArEstML'] = simsML['Ar_c']
                simsML['ArUnc'] = simsML['ArErr_c']

        # for later compatibility
        simsML['simple_single_Mr'] = simsML['MrEstML']
        simsML['simple_single_MrErr'] = simsML['MrUnc']
        simsML['simple_single_FeH'] = simsML['FeHEstML']
        simsML['simple_single_FeHErr'] = simsML['FeHUnc']
        simsML['simple_single_Ar'] = simsML['ArEstML']
        simsML['simple_single_ArErr'] = simsML['ArUnc']

        
        print(np.size(simsML), 'read from', file)
        return simsML


## finalized format for reading simCatalog files (after implementing 3D case with free Ar)              
def readKarloMLestimates3D(inKfile):
        print('READING FROM', inKfile)
        file = inKfile
        ## file includes:
        colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
        colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr']
        colnames = colnames + ['ugSL', 'grSL', 'riSL', 'izSL', 'ugErrSL', 'grErrSL', 'riErrSL', 'izErrSL'] 
        colnames = colnames + ['gi0', 'gi', 'test_set']
        colnames = colnames + ['simple_single_Mr', 'simple_single_MrErr', 'simple_single_Ar', 'simple_single_ArErr', 'simple_single_FeH', 'simple_single_FeHErr']
           
        ## read FeH and Mr estimates and fake their uncertainties 
        simsML = Table.read(file, format='ascii', names=colnames)
        simsML['MrEstML'] = simsML['simple_single_Mr']
        simsML['MrUnc'] = simsML['simple_single_MrErr']
        simsML['FeHEstML'] = simsML['simple_single_FeH']
        simsML['FeHUnc'] = simsML['simple_single_FeHErr']
             
        print(np.size(simsML), 'read from', file)
        return simsML
  
        
def LSSTsimsLocus(fixForStripe82=True): 
        ## Mr, as function of [Fe/H], along the SDSS/LSST stellar 
        ## for more details see the file header
        colnames = ['Mr', 'FeH', 'ug', 'gr', 'ri', 'iz', 'zy']
        LSSTlocus = Table.read('../data/LocusData/MSandRGBcolors_v1.3.txt', format='ascii', names=colnames)
        LSSTlocus['gi'] = LSSTlocus['gr'] + LSSTlocus['ri']
        if (fixForStripe82):
            print('Fixing input Mr-FeH-colors grid to agree with the SDSS v4.2 catalog')
            # for SDSS v4.2 catalog, see: http://faculty.washington.edu/ivezic/sdss/catalogs/stripe82.html
            # implement empirical corrections for u-g and i-z colors to make it better agree with the SDSS v4.2 catalog
            # fix u-g: slightly redder for progressively redder stars and fixed for gi>giMax
            ugFix = LSSTlocus['ug']+0.02*(2+LSSTlocus['FeH'])*LSSTlocus['gi']
            giMax = 1.8
            ugMax = 2.53 + 0.13*(1+LSSTlocus['FeH'])
            LSSTlocus['ug'] = np.where(LSSTlocus['gi']>giMax, ugMax , ugFix)
            # fix i-z color: small offsets as functions of r-i and [Fe/H]
            off0 = 0.08
            off2 = -0.09
            off5 = 0.008
            offZ = 0.01
            Z0 = 2.5
            LSSTlocus['iz'] += off0*LSSTlocus['ri']+off2*LSSTlocus['ri']**2+off5*LSSTlocus['ri']**5
            LSSTlocus['iz'] += offZ*(Z0+LSSTlocus['FeH'])
        return LSSTlocus

def BHBlocus(): 
        ## Mr, [Fe/H] and SDSS/LSST colors for BHB stars (Sirko+2004)
        ## [Fe/H] ranges from -2.50 to 0.0 in steps of 0.5 dex 
        colnames = ['Mr', 'FeH', 'ug', 'gr', 'ri', 'iz', 'zy']
        BHBlocus = Table.read('./BHBcolors_v1.0.dat', format='ascii', names=colnames)
        BHBlocus['gi'] = BHBlocus['gr'] + BHBlocus['ri'] 
        return BHBlocus

def photoFeH(ug, gr): 
        x = np.array(ug)
        y = np.array(gr)
        ## photometric metallicity introduced in Ivezic et al. (2008), Tomography II
        ## and revised in Bond et al. (2012), Tomography III (see Appendix A.1) 
        ## valid for SDSS bands and F/G stars defined by 
        ## 0.2 < g−r < 0.6 and −0.25 + 0.5*(u−g) < g−r < 0.05 + 0.5*(u−g)
        A, B, C, D, E, F, G, H, I, J = (-13.13, 14.09, 28.04, -5.51, -5.90, -58.68, 9.14, -20.61, 0.0, 58.20)
        return A + B*x + C*y + D*x*y + E*x**2 + F*y**2 + G*x**2*y + H*x*y**2 + I*x**3 + J*y**3 

def getMr(gi, FeH):
        ## Mr(g-i, FeH) introduced in Ivezic et al. (2008), Tomography II
        MrFit = -5.06 + 14.32*gi -12.97*gi**2 + 6.127*gi**3 -1.267*gi**4 + 0.0967*gi**5
        ## offset for metallicity, valid for -2.5 < FeH < 0.2
        FeHoffset = 4.50 -1.11*FeH -0.18*FeH**2
        return MrFit + FeHoffset

def extcoeff():
        ## coefficients to correct for ISM dust (for S82 from Berry+2012, Table 1)
        ## extcoeff(band) = A_band / A_r 
        extcoeff = {}
        extcoeff['u'] = 1.810
        extcoeff['g'] = 1.400
        extcoeff['r'] = 1.000  # by definition
        extcoeff['i'] = 0.759 
        extcoeff['z'] = 0.561 
        return extcoeff 
        
def getColorsFromMrFeH(L, Lvalues, colors=''):
    SDSScolors = ['ug', 'gr', 'ri', 'iz']
    if (colors == ''): colors = SDSScolors
    # grid properties
    MrMin = np.min(L['Mr'])
    MrMax = np.max(L['Mr'])
    dMr = L['Mr'][1]-L['Mr'][0]
    i = (Lvalues['Mr']-MrMin)/dMr
    i = i.astype(int)
    imax = (MrMax-MrMin)/dMr
    imax = imax.astype(int)
    FeHmin = np.min(L['FeH'])
    dFeH = L['FeH'][imax+2]-L['FeH'][0]
    j = (Lvalues['FeH']-FeHmin)/dFeH
    j = j.astype(int)
    k = i + j*(imax+2) + 1
    for color in colors:
        Lvalues[color] = L[color][k]
 
        
def getColorsFromGivenMrFeH(myMr, myFeH, L, gp, colors=""):
    SDSScolors = ['ug', 'gr', 'ri', 'iz'] 
    if (colors==""): colors = SDSScolors
    i = (myMr-gp['MrMin'])/gp['dMr']
    i = i.astype(int)
    j = (myFeH-gp['FeHmin'])/gp['dFeH']
    j = j.astype(int)
    k = i + j*(gp['imax']+2) + 1
    myColors = {}
    for color in colors:
        myColors[color] = L[color][k]
    return myColors
 
        
# given a grid of model colors, Mcolors, compute chi2
# for a given set of observed colors Ocolors, with errors Oerrors 
# colors to be used in chi2 computation are listed in colorNames
# Mcolors is astropy Table 
def getLocusChi2colors(colorNames, Mcolors, Ocolors, Oerrors):
    chi2 = 0*Mcolors[colorNames[0]]
    for color in colorNames:   
        chi2 += ((Ocolors[color]-Mcolors[color])/Oerrors[color])**2 
    return chi2


def logLfromObsColors(Mr, FeH, ObsColors, ObsColErrors, gridProperties, SDSScolors):
    myModelColors = getColorsFromGivenMrFeH(Mr, FeH, LSSTlocus, gridProperties)
    chi2 = getLocusChi2colors(SDSScolors, myModelColors, ObsColors, ObsColErrors)
    # note that logL = -1/2 * chi2 (for gaussian errors)
    return (-0.5*chi2)


## for testing procedures 
def getFitResults(i, gridStride, Ntest, colors, locusGrid, ObsColorErr, locusFitMS, locusFitRG, results):
    tempResults = {}
    for Q in ('chi2', 'Mr', 'FeH'):
        for Stype in ('MS', 'RG'):
            tempResults[Q, Stype] = np.zeros(Ntest)   
    
    for j in range(0,Ntest):
        #print('       test #', j)
        ObsColor = {}
        for color in colors:
            ## this is where random noise is generated and added to input magnitudes ##
            ObsColor[color] = locusGrid[color][i*gridStride] + np.random.normal(0, ObsColorErr[color])
            # print(color, locusFeH[color][i], ObsColor[color], ObsColorErr[color])
        ## chi2 for each grid point
        chi2MS = getLocusChi2colors(colors, locusFitMS, ObsColor, ObsColorErr)
        chi2RG = getLocusChi2colors(colors, locusFitRG, ObsColor, ObsColorErr)
        ## store Mr and FeH values corresponding to the minimum chi2
        # MS: 
        kmin = np.argmin(chi2MS)
        tempResults['chi2', 'MS'][j] = chi2MS[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'MS'][j] = locusFitMS[Q][kmin] 
        # RG: 
        kmin = np.argmin(chi2RG)
        tempResults['chi2', 'RG'][j] = chi2RG[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'RG'][j] = locusFitRG[Q][kmin] 

    for Q in ('chi2', 'Mr', 'FeH'):
        results[Q, 'MSmean'][i] = np.mean(tempResults[Q,'MS'])
        results[Q, 'MSstd'][i] = np.std(tempResults[Q, 'MS'])
        results[Q, 'RGmean'][i] = np.mean(tempResults[Q,'RG'])
        results[Q, 'RGstd'][i] = np.std(tempResults[Q, 'RG']) 
    return 

# given chi2 values for every grid point, return the grid point with the smallest chi2
def extractBestFit(chi2vals, locusGrid):
    kmin = np.argmin(chi2vals)
    return locusGrid[kmin], kmin


## similar to getFitResults, but for fitting a sample where each entry has DIFFERENT color errors
def fitPhotoD(i, gridStride, colors, data2fit, locusFitMS, locusFitRG, results):
    Ntest = 1
    tempResults = {}
    for Q in ('chi2', 'Mr', 'FeH'):
        for Stype in ('MS', 'RG'):
            tempResults[Q,Stype] = np.zeros(Ntest)   
    
    for j in range(0,Ntest):
        #print('       test #', j)
        ObsColor = {}
        ObsColorErr = {}
        for color in colors:
            errname = color + 'Err'
            ObsColorErr[color] = data2fit[errname][i*gridStride]
            if (0): 
                ## this is where random noise is generated and added to input magnitudes ##
                ObsColor[color] = data2fit[color][i*gridStride] + np.random.normal(0, ObsColorErr[color])
                # print(color, locusFeH[color][i], ObsColor[color], ObsColorErr[color])
            else:
                ## assuming that colors already have random noise and are corrected for extinction 
                ObsColor[color] = data2fit[color][i*gridStride] 
        ## chi2 for each grid point
        chi2MS = getLocusChi2colors(colors, locusFitMS, ObsColor, ObsColorErr)
        chi2RG = getLocusChi2colors(colors, locusFitRG, ObsColor, ObsColorErr)
        ## store Mr and FeH values corresponding to the minimum chi2
        # MS: 
        kmin = np.argmin(chi2MS)
        tempResults['chi2', 'MS'][j] = chi2MS[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'MS'][j] = locusFitMS[Q][kmin] 
        # RG: 
        kmin = np.argmin(chi2RG)
        tempResults['chi2', 'RG'][j] = chi2RG[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'RG'][j] = locusFitRG[Q][kmin]

        ## NB: above needs to be changed so that instead of minimum chi2
        ##     and no scatter estimate, all chi2 < chi2max are selected
        ##     and then 2D Gauss is fit (as in Berry+2012)
        ##     right not std is 0 for all cases 

    for Q in ('chi2', 'Mr', 'FeH'):
        results[Q, 'MSmean'][i] = np.mean(tempResults[Q,'MS'])
        results[Q, 'MSstd'][i] = np.std(tempResults[Q, 'MS'])
        results[Q, 'RGmean'][i] = np.mean(tempResults[Q,'RG'])
        results[Q, 'RGstd'][i] = np.std(tempResults[Q, 'RG']) 
    return 



## similar to fitPhotoD, but simply returning chi2 mag for one isochrone family, without any processing 
def getPhotoDchi2map(i, colors, data2fit, locus):

        #print('getPhotoDchi2map: colors=', colors)
        # set up colors for fitting
        ObsColor = {}
        ObsColorErr = {}
        for color in colors:
            # print('    color=', color)
            errname = color + 'Err'
            ObsColorErr[color] = data2fit[errname][i]
            # assuming that colors already have random noise and are corrected for extinction 
            ObsColor[color] = data2fit[color][i]
            
        ## return chi2map for each grid point
        return getLocusChi2colors(colors, locus, ObsColor, ObsColorErr)



## similar to getPhotoDchi2map, but adding the 3rd dimension (Ar) to model (locus) colors
## the upper limit on Ar comes from an external Ar (e.g. ArSFD for obs, or used Ar in sims)
## NB the grid for Ar is defined here 
def getPhotoDchi2map3D(i, colors, colorReddCoeffs, data2fit, locus, ArCoeff, masterLocus=False):

        # first adopt, or generate, 3D model locus
        if masterLocus:
            locus3D = locus 
        else:
            # extend 2D Mr-FeH grid in zero-reddening locus (astropy Table), to a 3D color grid by
            # adding reddening grid to each entry in locus (which corresponds to ArGrid[0] = 0)
            ArMax = ArCoeff[0]*data2fit['Ar'][i] + ArCoeff[1]
            nArGrid = int(ArMax/ArCoeff[2]) + 1
            if (nArGrid>1000):
                print('resetting nArGrid to 1000 in getPhotoDchi2map3D, from:', nArGrid)
                nArGrid = 1000
            if (1):
                ArGrid = np.linspace(0, ArMax, nArGrid)
            else:
                # this is for testing performance when Ar prior is delta function centered on true value
                ArGrid = np.linspace(data2fit['Ar'][i], data2fit['Ar'][i], 1)

            # color corrections due to dust reddening (for each Ar in the grid for this particular star) 
            colorCorrection = {}
            for color in colors:
               colorCorrection[color] = ArGrid * colorReddCoeffs[color]
            locus3D = make3Dlocus(locus, ArGrid, colors, colorCorrection)
        
        # set up colors for fitting (for this star specified by input "i") 
        ObsColor = {}
        ObsColorErr = {}
        for color in colors:
            # print('    color=', color)
            ObsColor[color] = data2fit[color][i]
            errname = color + 'Err'
            ObsColorErr[color] = data2fit[errname][i]

        ## return chi2map (data cube) for each grid point in locus3D 
        if masterLocus:
            return getLocusChi2colors(colors, locus3D, ObsColor, ObsColorErr)
        else:
            return ArGrid, getLocusChi2colors(colors, locus3D, ObsColor, ObsColorErr)


def get3DmodelList(locusData, fitColors, agressive=False):


    if agressive:
        ## AGRESSIVE 
        # for small 3D locus: 
        ArGridSmall = np.linspace(0,0.5,101)   # step 0.005 mag
        # for medium 3D locus: 
        ArGridMedium = np.linspace(0,2.0,201)  # step 0.01 mag
        # for large 3D locus: 
        ArGridLarge = np.linspace(0,5.0,251)   # step 0.02 mag
    else:
        ## LESS AGRESSIVE
        ArGridSmall = np.linspace(0,0.3,31)    # step 0.01 mag
        ArGridMedium = np.linspace(0,0.8,81)   # step 0.01 mag
        ArGridLarge = np.linspace(0,2.5,126)   # step 0.02 mag

    AGList = []
    AGList.append(ArGridSmall)
    AGList.append(ArGridMedium)
    AGList.append(ArGridLarge)

    ### call the workhorse 
    L3Dlist = make3DlocusList(locusData, fitColors, AGList)
    
    # repack
    ArGridList = {}
    locus3DList = {}
    locus3DList['ArSmall'] = L3Dlist[0]
    ArGridList['ArSmall'] = ArGridSmall
    locus3DList['ArMedium'] = L3Dlist[1] 
    ArGridList['ArMedium'] = ArGridMedium
    locus3DList['ArLarge'] = L3Dlist[2] 
    ArGridList['ArLarge'] = ArGridLarge
    return ArGridList, locus3DList



### make 3Dlocus list for provided list of Ar grids
def make3DlocusList(locusData, fitColors, ArGridList):

    # color corrections due to dust reddening 
    # for finding extinction, too
    C = extcoeff()
    reddCoeffs = {}
    reddCoeffs['ug'] = C['u']-C['g']
    reddCoeffs['gr'] = C['g']-C['r']
    reddCoeffs['ri'] = C['r']-C['i']
    reddCoeffs['iz'] = C['i']-C['z']

    # intrinsic table sizes
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    
    # turn astropy table into numpy array
    locusData['Ar'] = 0*locusData['Mr']
    LocusNP = np.array(locusData) 
    # the repeating block 
    locus3D0 = LocusNP.reshape(np.size(FeH1d), np.size(Mr1d))

    locus3DList = []
    for ArGrid in ArGridList:
        colCorr = {}
        for color in fitColors:
            colCorr[color] = ArGrid * reddCoeffs[color]
        locus3D = make3DlocusFast(locus3D0, ArGrid, fitColors, colCorr, FeH1d, Mr1d)
        locus3DList.append(locus3D)
    return locus3DList 


## VOLATILE: assumes order of colors in locus3D0 (that must be consistent with colCorr
##      NB IT WILL BREAK WHEN ANOTHER COLOR IS ADDED!  (e.g. z-y for LSST data) 
## given 2D numpy array, make a 3D numpy array by replicating it for each element 
## in ArGrid and apply reddening corrections
## n.b. colors is not used (place holder to fix VOLATILE problem...)
def make3DlocusFast(locus3D0, ArGrid, colors, colorCorrection, FeH1d, Mr1d):

    N3rd = np.size(ArGrid)
    locus3D = np.repeat(locus3D0[:, :, np.newaxis], N3rd, axis=2)
    for i in range(0,np.size(FeH1d)):
        for j in range(0,np.size(Mr1d)):
            for k in range(0,np.size(ArGrid)):
                locus3D[i,j,k][2] = locus3D[i,j,k][2] + colorCorrection['ug'][k] 
                locus3D[i,j,k][3] = locus3D[i,j,k][3] + colorCorrection['gr'][k] 
                locus3D[i,j,k][4] = locus3D[i,j,k][4] + colorCorrection['ri'][k] 
                locus3D[i,j,k][5] = locus3D[i,j,k][5] + colorCorrection['iz'][k] 
                locus3D[i,j,k][8] = ArGrid[k]   
    return locus3D 


### WHY IS THIS CODE SCALING WITH THE SQUARE OF ArGrid LENGTH???    
# replace each row in locus (astropy Table) with np.size(ArGrid) rows where colors in colors
# are reddened using the values in colCorr and return the resulting astropy Table
def make3Dlocus(locus, ArGrid, colors, colCorr):
    
    # initialize the first block of 3D table that corresponds to Ar=0 and the input table 
    locus3D = Table((locus['Mr'], locus['FeH']), copy=True)
    for color in colors: locus3D.add_column(locus[color])
    # the first point in Ar grid is usually, but NOT necessarily, equal to 0
    locus3D['Ar'] = 0*locus3D['Mr'] + ArGrid[0]
    for color in colors:
        locus3D[color] = locus3D[color] + colCorr[color][0]
    
    # loop over all >0 reddening values 
    for k in range(1, np.size(ArGrid)):
        # new block, start with a copy of the input table
        locusAr = Table((locus['Mr'], locus['FeH']), copy=True)
        # add a column with the corresponding value of Ar
        locusAr['Ar'] = 0*locusAr['Mr'] + ArGrid[k]
        # and now redden zero-reddening colors with provided reddening corrections 
        cRed = {}
        for color in colors:
            cRed[color] = locus[color] + colCorr[color][k]
            locusAr.add_column(cRed[color])
        # now vstack the segment for this Ar value to locus3D table:  
        locus3D = vstack([locus3D, locusAr])
 
    return locus3D 

## subsample locusData along Mr and FeH grids by factors kMr and kFeH (if both are 1, no subsampling)
def subsampleLocusData(locusData, kMr, kFeH, verbose=True):
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    # original grid sizes
    nFeH = FeH1d.size
    nMr = Mr1d.size
    # subsampled grid sizes
    nFeHs = int(nFeH/kFeH)
    nMrs = int(nMr/kMr)
    if verbose: print('subsampled locus 2D grid in FeH and Mr from', nFeH, nMr, 'to:', nFeHs, nMrs)
    subsampled = locusData[:0].copy()
    # now add subsampled rows from the input table
    for j in range(0,nFeHs):
        for i in range(0,nMrs):
            k = i*kMr + j*kFeH*nMr
            subsampled.add_row(locusData[k]) 
    return subsampled


#########
## some statistics helpers...

# given vectors x and y, fit medians in bins from xMin to xMax, with Nbin steps,
# and return xBin, medianBin, medianErrBin 
def fitMedians(x, y, xMin, xMax, Nbin, verbose=False): 

    # first generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    xBin = np.linspace(0, 1, Nbin)
    nPts = 0*np.linspace(0, 1, Nbin)
    medianBin = 0*np.linspace(0, 1, Nbin)
    sigGbin = -1+0*np.linspace(0, 1, Nbin) 
    for i in range(0, Nbin): 
        xBin[i] = 0.5*(xEdge[i]+xEdge[i+1]) 
        yAux = y[(x>xEdge[i])&(x<=xEdge[i+1])]
        if (yAux.size > 0):
            nPts[i] = yAux.size
            medianBin[i] = np.median(yAux)
            # from scipy import stats
            # medianBin[i] = stats.mode(yAux)[0][0]
            # medianBin[i] = np.mean(yAux)
            # robust estimate of standard deviation: 0.741*(q75-q25)
            sigmaG = 0.741*(np.percentile(yAux,75)-np.percentile(yAux,25))
            # uncertainty of the median: sqrt(pi/2)*st.dev/sqrt(N)
            sigGbin[i] = np.sqrt(np.pi/2)*sigmaG/np.sqrt(nPts[i])
        else:
            nPts[i] = 0 
            medianBin[i] = 0 
            sigGbin[i] = 0 
            # nPts[i], medianBin[i], sigGBin[i] = 0 
        
    if (verbose):
        print('median:', np.median(medianBin[nPts>0]))

    return xBin, nPts, medianBin, sigGbin


def basicStats(df, colName):
    yAux = df[colName]
    # robust estimate of standard deviation: 0.741*(q75-q25)
    sigmaG = 0.741*(np.percentile(yAux,75)-np.percentile(yAux,25))
    median = np.median(yAux)
    return [np.size(yAux), np.min(yAux), np.max(yAux), median, sigmaG]


def getMedianSigG(basicStatsLine):
    med = "%.3f" % basicStatsLine[3]
    sigG = "%.2f" % basicStatsLine[4]
    return [med, sigG, basicStatsLine[0]]

def makeStatsTable3D(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):

    # first mag threshold
    dfm = df[df[magName]<=magThresh]
    
    # split along Mr sequence: giants, main-sequence blue and red stars
    dfG = dfm[dfm['Mr']<=Mr1]
    dfB = dfm[(dfm['Mr']>Mr1)&(dfm['Mr']<=Mr2)]
    dfR = dfm[dfm['Mr']>Mr2]
    
    # and finally split by metallicity, both because FeH sensitivity to u-g, and because halo vs. disk distinction
    #    "h" is for "halo", not "high" 
    dfGh = dfG[dfG['FeH']<=FeHthresh]
    dfGd = dfG[dfG['FeH']>FeHthresh]
    dfBh = dfB[dfB['FeH']<=FeHthresh]
    dfBd = dfB[dfB['FeH']>FeHthresh]
    dfRh = dfR[dfR['FeH']<=FeHthresh]
    dfRd = dfR[dfR['FeH']>FeHthresh]

    # and for all (without Mr split):
    dfh = dfm[dfm['FeH']<=FeHthresh]
    dfd = dfm[dfm['FeH']>FeHthresh]
 
    print('---------------------------------------------------------------------------------------------------')
    print('      FULL SAMPLE:          ', dMrname, '                    ', dFeHname, '                        Ar')
    print('             all:', getMedianSigG(basicStats(df, dMrname)), getMedianSigG(basicStats(df, dFeHname)), getMedianSigG(basicStats(df, 'dAr')))
    print('    mag selected:', getMedianSigG(basicStats(dfm, dMrname)), getMedianSigG(basicStats(dfm, dFeHname)), getMedianSigG(basicStats(dfm, 'dAr')))
    print('---------------------------------------------------------------------------------------------------')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfh, dMrname)), getMedianSigG(basicStats(dfh, dFeHname)), getMedianSigG(basicStats(dfh, 'dAr')))
    print('          giants:', getMedianSigG(basicStats(dfGh, dMrname)), getMedianSigG(basicStats(dfGh, dFeHname)), getMedianSigG(basicStats(dfGh, 'dAr')))
    print('         blue MS:', getMedianSigG(basicStats(dfBh, dMrname)), getMedianSigG(basicStats(dfBh, dFeHname)), getMedianSigG(basicStats(dfBh, 'dAr')))
    print('          red MS:', getMedianSigG(basicStats(dfRh, dMrname)), getMedianSigG(basicStats(dfRh, dFeHname)), getMedianSigG(basicStats(dfRh, 'dAr')))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfd, dMrname)), getMedianSigG(basicStats(dfd, dFeHname)), getMedianSigG(basicStats(dfd, 'dAr')))
    print('          giants:', getMedianSigG(basicStats(dfGd, dMrname)), getMedianSigG(basicStats(dfGd, dFeHname)), getMedianSigG(basicStats(dfGd, 'dAr')))
    print('         blue MS:', getMedianSigG(basicStats(dfBd, dMrname)), getMedianSigG(basicStats(dfBd, dFeHname)), getMedianSigG(basicStats(dfBd, 'dAr')))
    print('          red MS:', getMedianSigG(basicStats(dfRd, dMrname)), getMedianSigG(basicStats(dfRd, dFeHname)), getMedianSigG(basicStats(dfRd, 'dAr')))
    print('---------------------------------------------------------------------------------------------------')

    return


def makeStatsTable0(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):

    # first mag threshold
    dfm = df[df[magName]<=magThresh]
    
    # split along Mr sequence: giants, main-sequence blue and red stars
    dfG = dfm[dfm['Mr']<=Mr1]
    dfB = dfm[(dfm['Mr']>Mr1)&(dfm['Mr']<=Mr2)]
    dfR = dfm[dfm['Mr']>Mr2]
    
    # and finally split by metallicity, both because FeH sensitivity to u-g, and because halo vs. disk distinction
    #    "h" is for "halo", not "high" 
    dfGh = dfG[dfG['FeH']<=FeHthresh]
    dfGd = dfG[dfG['FeH']>FeHthresh]
    dfBh = dfB[dfB['FeH']<=FeHthresh]
    dfBd = dfB[dfB['FeH']>FeHthresh]
    dfRh = dfR[dfR['FeH']<=FeHthresh]
    dfRd = dfR[dfR['FeH']>FeHthresh]

    # and for all (without Mr split):
    dfh = dfm[dfm['FeH']<=FeHthresh]
    dfd = dfm[dfm['FeH']>FeHthresh]
 
    print('---------------------------------------------------------------------')
    print('      FULL SAMPLE:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(df, dMrname)), getMedianSigG(basicStats(df, dFeHname)))
    print('    mag selected:', getMedianSigG(basicStats(dfm, dMrname)), getMedianSigG(basicStats(dfm, dFeHname)))
    print('---------------------------------------------------------------------')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfh, dMrname)), getMedianSigG(basicStats(dfh, dFeHname)))
    print('          giants:', getMedianSigG(basicStats(dfGh, dMrname)), getMedianSigG(basicStats(dfGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBh, dMrname)), getMedianSigG(basicStats(dfBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfRh, dMrname)), getMedianSigG(basicStats(dfRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfd, dMrname)), getMedianSigG(basicStats(dfd, dFeHname)))
    print('          giants:', getMedianSigG(basicStats(dfGd, dMrname)), getMedianSigG(basicStats(dfGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBd, dMrname)), getMedianSigG(basicStats(dfBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfRd, dMrname)), getMedianSigG(basicStats(dfRd, dFeHname)))
    print('---------------------------------------------------------')

    return


def makeStatsTable(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):
    # first split by SNR 
    dfBright = df[df[magName]<=magThresh]
    dfFaint = df[df[magName]>magThresh]
    # then split along Mr sequence: giants, main-sequence blue and red stars
    dfBrightG = dfBright[dfBright['Mr']<=Mr1]
    dfBrightB = dfBright[(dfBright['Mr']>Mr1)&(dfBright['Mr']<=Mr2)]
    dfBrightR = dfBright[dfBright['Mr']>Mr2]
    dfFaintG = dfFaint[dfFaint['Mr']<=Mr1]
    dfFaintB = dfFaint[(dfFaint['Mr']>Mr1)&(dfFaint['Mr']<=Mr2)]
    dfFaintR = dfFaint[dfFaint['Mr']>Mr2]
    # and finally split by metallicity, both because FeH sensitivity to u-g, and because halo vs. disk distinction
    dfBrightGh = dfBrightG[dfBrightG['FeH']<=FeHthresh]
    dfBrightGd = dfBrightG[dfBrightG['FeH']>FeHthresh]
    dfBrightBh = dfBrightB[dfBrightB['FeH']<=FeHthresh]
    dfBrightBd = dfBrightB[dfBrightB['FeH']>FeHthresh]
    dfBrightRh = dfBrightR[dfBrightR['FeH']<=FeHthresh]
    dfBrightRd = dfBrightR[dfBrightR['FeH']>FeHthresh]
    dfFaintGh = dfFaintG[dfFaintG['FeH']<=FeHthresh]
    dfFaintGd = dfFaintG[dfFaintG['FeH']>FeHthresh]
    dfFaintBh = dfFaintB[dfFaintB['FeH']<=FeHthresh]
    dfFaintBd = dfFaintB[dfFaintB['FeH']>FeHthresh]
    dfFaintRh = dfFaintR[dfFaintR['FeH']<=FeHthresh]
    dfFaintRd = dfFaintR[dfFaintR['FeH']>FeHthresh]

    print('---------------------------------------------------------------------')
    print(' -- high SNR  ---')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfBrightGh, dMrname)), getMedianSigG(basicStats(dfBrightGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBrightBh, dMrname)), getMedianSigG(basicStats(dfBrightBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfBrightRh, dMrname)), getMedianSigG(basicStats(dfBrightRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfBrightGd, dMrname)), getMedianSigG(basicStats(dfBrightGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBrightBd, dMrname)), getMedianSigG(basicStats(dfBrightBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfBrightRd, dMrname)), getMedianSigG(basicStats(dfBrightRd, dFeHname)))
    print(' --  low SNR  ---')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfFaintGh, dMrname)), getMedianSigG(basicStats(dfFaintGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfFaintBh, dMrname)), getMedianSigG(basicStats(dfFaintBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfFaintRh, dMrname)), getMedianSigG(basicStats(dfFaintRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfFaintGd, dMrname)), getMedianSigG(basicStats(dfFaintGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfFaintBd, dMrname)), getMedianSigG(basicStats(dfFaintBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfFaintRd, dMrname)), getMedianSigG(basicStats(dfFaintRd, dFeHname)))
    print('---------------------------------------------------------')

    return





def getLSSTm5(data, depth='coadd', magVersion=False, suffix=''):
    # temporary: only use SDSS colors
    bandpasses = ['u', 'g', 'r', 'i', 'z']
    # from https://iopscience.iop.org/article/10.3847/1538-4365/ac3e72
    coaddm5 = {}
    coaddm5['u'] = 25.73
    coaddm5['g'] = 26.86
    coaddm5['r'] = 26.88
    coaddm5['i'] = 26.34
    coaddm5['z'] = 25.63
    coaddm5['y'] = 24.87
    singlem5 = {}
    singlem5['u'] = 23.50
    singlem5['g'] = 24.44 
    singlem5['r'] = 23.98 
    singlem5['i'] = 23.41
    singlem5['z'] = 22.77
    singlem5['y'] = 22.01
    gg = {}
    gg['u'] = 0.038 
    gg['g'] = 0.039 
    gg['r'] = 0.039 
    gg['i'] = 0.039 
    gg['z'] = 0.039 
    gg['y'] = 0.039   
    m5 = {}
    for b in bandpasses:
        if (depth=='coadd'):
            m5[b] = coaddm5[b] 
        else:
            m5[b] = singlem5[b] 
    mags = {}
    for b in bandpasses:
        if (magVersion):
            mags[b] = data[b+'mag'+suffix]
        else:
            mags[b] = data[b]
    errors = {}
    for b in bandpasses:
        x = 10**(0.4*(mags[b]-m5[b]))
        errors[b] = np.sqrt(0.005**2 + (0.04-gg[b])*x + gg[b]*x**2)
    return errors

### this inverts the error(mag) relation from getLSSTm5 and returns errors for provided magnitudes
### N.B. getLSSTm5 also assumes SDSS bandpasses (that is, no y band) 
def getLSSTm5err(mags, depth='coadd'):
    # temporary: only use SDSS colors (no y band)
    bandpasses = ['u', 'g', 'r', 'i', 'z']
    # arrays for interpolation
    magGrid = np.linspace(10, 30, 2001)  # 0.01 mag steps
    magData = {}
    for b in bandpasses:
        magData[b] = magGrid
    errGrid = getLSSTm5(magData, depth)
    # now interpolate to get errors 
    errors = {}
    for b in bandpasses:
        errors[b] = np.interp(mags[b], magGrid, errGrid[b]) 
    return errors
