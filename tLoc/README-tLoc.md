
== ZI summary about how to use tLoc instead of the initial Mr parametrization of SED models (July 20, '26) ==

Here, tLoc is a running variable along the locus, design to prevent instances of locus having two
colors for a given Mr (after turnoff, the sequence of Mr vs. color sometimes turns "down" towards
fainter Mr than turn-off Mr). 

1) tLoc is defined in augmentSDSSlocusWithIsochrone from LocusTools-tLoc.py

After staring at the code for a while, and thinking about what I was thinking 2-3 years ago, 
I copied the code to ChatGPT and it gave a decent description of it. I sent you email on July 20,
2026 with a Subject: tLoc definition 

E.g.: 
# tLoc     Mr     FeH      ug         gr         ri           iz 
# 3.53   3.06 -1.90   0.854   0.225   0.071  -0.007
# 4.15   4.15 -1.80   0.828   0.231   0.070  -0.009


2) changing the code to use tLoc

The basic idea is to replace Mr by tLoc when running the Bayes code, and then at the end
project the best tLoc and its uncertainty back to Mr axis.

In notebook: tLoc/exampleBayesMethod3D-tLoc.ipynb
(this notebook is a copy of a notebook that Lovro used for testing:
exampleBayesMethod3D_SDSSpatchRA340-350-simLSSTcatalog_LovroGPUtest.ipynb) 

First, overwrite Mr by tLoc values: 

### hack to switch to tLoc implementation
OKlocus['MrTrue'] = OKlocus['Mr']
OKlocus['Mr'] = OKlocus['tLoc']

and then after the Bayes step/computation is done, go back: 

Look for section title "correct for Mr < 4 projection of Mr to tLoc" and cells and plots after it. 

The key call is to function getMrFromFeHtLoc from LocusTools-tLoc.py 
That function is a most horrible despicable hack but it works, look for section title 
"correct for tLoc vs. Mr mapping" and plots after it. 

It would be great if Lovorka could reproduce these steps on an example with simulated stars
from TRILEGAL, where we know the correct answers... Then these two ugly hacks of overwriting 
Mr by tLoc, and especially getMrFromFeHtLoc should be reimplemented in a better way. 
 

3) for plotting:

compare2isochronesColorMrAlongLocus in paperPlots-tLoc.py takes two SEDs (data frames) and 
plots colors vs. tLoc. 

Note also that tLoc/exampleBayesMethod3D-tLoc.ipynb has a few nice QA plots at the end. 


=============================================================
NOTES from ZI to ZI (not directly relevant for discussion above, but related): 

- temporary DP2-COSMOS catalog, PhotoD-COSMOSstars.parquet, 
    used to generate new models, LSSTlocus_1Gyr.txt and LSSTlocus_10Gyr.txt,
	was made with code in 
   SGseparation/star-galaxy-separationZI/notebooks/COSMOS-PhotoD-maketest1.ipynb
    and starting from data in SGseparation/star-galaxy-DP2-data/DP2_COSMOS_objects.fits

- using PhotoD-COSMOSstars.parquet, new SED models are generated with code in 
   SGseparation/star-galaxy-separationZI/notebooks/COSMOS-PhotoD-compare2models.ipynb
   ALSO: it makes RubinStellarLocus_COSMOS.txt file! 
   
- tLoc parametrization in LSSTlocus_10Gyr.txt etc is adopted from SDSSDSEDlocus_10Gyr.txt 

- SDSSDSEDlocus_10Gyr.txt etc are produced with 
   PhotoD/PhotoD/notebooks/makeAugmentedLocusTable.ipynb

   starting from PhotoD/PhotoD/data/LocusData/MSandRGBcolors_v1.3.txt 
       and using DSED models 
   see also plots in PhotoD/PhotoD/notebooks/augmentLocusTable.ipynb
=============================================================


