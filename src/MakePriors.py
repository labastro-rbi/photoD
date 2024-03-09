### tools to support the production of metadata and maps with priors based on TRILEGAL simulations

## import supporting local tools: 
import LocusTools as lt
import BayesTools as bt
import numpy as np
from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert
from getpass import getpass

## define query with RA/Dec boundaries and a list of quantities to retrieve from lsst_sim.simdr2 table at Astro Lab 
# TRILEGAL NOTE:
# gc depends on which Galactic component the star belongs to: 
#    1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
# label depends on the evolutionary phase of the star, as:
# 0 pre-main sequence
# 1 main sequence
# 2 Hertzsprung gap
# 3 red giants
# 4–6 core He burning 
# 7 EAGB
# 8 TPAGB
# 9 Post-AGB
# 10 CO-WD 
# 21 22 23 24 25 26 27 He, NS, BH
def makeQueryString(minRA, maxRA, minDec, maxDec, umagMax=99.99):
    query1 = """
    SELECT ra, dec, gall, galb, gc, logage, mass, label, logg, m_h, av, mu0, umag, gmag, rmag, imag, zmag, ymag
    FROM lsst_sim.simdr2
    WHERE"""
    query2 = f" {minRA} < ra and ra < {maxRA}"
    query3 = f" AND {minDec} < dec and dec < {maxDec}"
    query4 = f" AND umag < {umagMax}"
    query = query1 + query2 + query3 + query4 + " \n"
    return query

## given a query, retrieve data from the database
def retrievePatch(query, verbose=True):
    # similar to processPatch() below but adds ugiz magnitudes
    if verbose: print('querying...')
    result = qc.query(sql=query)
    if verbose: print('converting...')
    trilegal = convert(result)
    trilegal['Ar'] = 2.75 * trilegal['av'] / 3.10 
    trilegal['Mr'] = trilegal['rmag'] - trilegal['Ar'] - trilegal['mu0']
    trilegal2 = trilegal.rename(columns={"m_h": "FeH"})
    df = trilegal2.rename(columns={"mu0": "DM"})
    # colors corrected for extinction
    C = lt. extcoeff()
    for b in ['u', 'g', 'r', 'i', 'z']:
        # correcting magnitudes for extinction 
        df[b] = df[b+'mag'] - C[b]*df['Ar'] 
    # correcting colors for extinction effects 
    df['ug'] = df['u'] - df['g'] 
    df['gr'] = df['g'] - df['r'] 
    df['ri'] = df['r'] - df['i'] 
    df['iz'] = df['i'] - df['z'] 
    df['gi'] = df['gr'] + df['ri']    
    if verbose: print('retrieved patch with', len(df), ' entries')
    return df

## given a query, dump maps with priors and metadata summary
def dumpPriorsFromPatch(df, rootname, show2Dmap=True, verbose=True):
    # NB map parameters are set in dumpPriorMaps(), by calling getBayesConstants()
    if verbose: print('dumping maps...')
    bt.dumpPriorMaps(df, rootname, show2Dmap=show2Dmap, verbose=verbose)
    print('done with', rootname)
    return

def printGalacticComponentStats(df, verbose=True):
    C1 = df[df['gc']==1]
    C2 = df[df['gc']==2]
    C3 = df[df['gc']==3]
    C4 = df[df['gc']==4]
    C5 = df[df['gc']==5]
    Ctotal = len(df)
    if verbose: print('Total:', Ctotal, 'Components 1-5:', len(C1), len(C2), len(C3), len(C4), len(C5))
    if verbose: print('           1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds') 
    return Ctotal, len(C1)/Ctotal, len(C2)/Ctotal, len(C3)/Ctotal, len(C4)/Ctotal, len(C5)/Ctotal

def printPopulationStats(df, verbose=True):
    T0 = df[df['label']==0]
    T1 = df[df['label']==1]
    T2 = df[df['label']==2]
    T3 = df[df['label']==3]
    T456 = df[(df['label']>3)&(df['label']<7)]
    T7 = df[df['label']==7]
    T8 = df[df['label']==8]
    T9 = df[df['label']==9]
    ## counts for isochrone stats
    # main-sequence stars
    Tms = len(T1)  
    # post-main-sequence stars
    Tpms = len(T2) + len(T3) + len(T456)
    # counts for AGB 
    Tagb = len(T7) + len(T8)
    # white dwarf counts
    Twd = len(T9)
    # probabilities
    Ttotal = Tms + Tpms + Tagb + Twd
    if (Ttotal>0):
        pms = Tms/Ttotal
        ppms = Tpms/Ttotal
        pagb = Tagb/Ttotal    
        pwd = Twd/Ttotal
    else: pms = ppms = pagb = pwd = -1
    if verbose: print('Total:', len(df), 'Populations:', len(T0), len(T1), len(T2), len(T3), len(T456), len(T7), len(T8), len(T9))
    if verbose: print('    Evolutionary phases: PMS=0, MS=1, SGB=2, RGB=3, CHeB=4,5,6, EAGB=7, TPAGB=8, PAGB+WD=9')
    if verbose: print(' Probabilities for MS, pMS, AGB and WD:', pms, ppms, pagb, pwd)  
    return Ttotal, pms, ppms, pagb, pwd


## dump catalog with retrieved data using above query and retrievePatch below
def dumpCatalogFromPatch(df, outfilename, verbose=True):
    if verbose: print('dumping catalog...')
    fout = open(outfilename, "w")
    # coordinates
    fout.write("    ra           dec           glon         glat")
    # parameters
    fout.write("      GC logage  mass   pop  logg   FeH      Mr     Ar    DM   ")
    # raw magnitudes
    fout.write("    umag    gmag    rmag    imag    zmag    ymag")
    # dereddened colors
    fout.write("     ug      gr      ri      iz      gi   \n")
    for i in range(0,len(df)):
        # coordinates
        r1 = df['ra'][i]
        r2 = df['dec'][i]
        r3 = df['gall'][i]
        r4 = df['galb'][i]
        s = str("%12.8f " % r1) + str("%12.8f  " % r2) + str("%12.8f " % r3) + str("%12.8f  " % r4)  
        # parameters 
        r1 = df['gc'][i]
        r2 = df['logage'][i]
        r3 = df['mass'][i]
        r4 = df['label'][i]
        r5 = df['logg'][i]
        r6 = df['FeH'][i]
        r7 = df['Mr'][i]
        r8 = df['Ar'][i]  
        r9 = df['DM'][i]
        s = s + str("%2.0f " % r1) + str("%5.2f  " % r2) + str("%6.3f  " % r3) + str("%2.0f  " % r4)
        s = s + str("%5.3f " % r5) + str("%6.2f  " % r6) + str("%6.2f  " % r7) + str("%5.2f  " % r8) + str("%5.2f  " % r9)
        # magnitudes (not corrected for ISM extinction)
        r1 = df['umag'][i]
        r2 = df['gmag'][i]
        r3 = df['rmag'][i]
        r4 = df['imag'][i]
        r5 = df['zmag'][i]
        r6 = df['ymag'][i]
        s = s + str("%8.3f" % r1) + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5) + str("%8.3f" % r6) 
        # colors, corrected for ISM extinction 
        r1 = df['ug'][i]
        r2 = df['gr'][i] 
        r3 = df['ri'][i] 
        r4 = df['iz'][i] 
        r5 = df['gi'][i] 
        s = s + str("%8.3f" % r1) + str("%8.3f" % r2) + str("%8.3f" % r3) + str("%8.3f" % r4) + str("%8.3f" % r5)
        s = s + "\n"
        fout.write(s)             
    fout.close() 
    if verbose: print('made catalog:', outfilename)
    return 
