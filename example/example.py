import os,sys
import numpy as np
import time

from astropy.io import fits, ascii
from astropy.table import Table, hstack, vstack
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

# import JPP
sys.path.append("../code/")
import jpp as jpp

# nice Style sheet
plt.style.use('../code/main.mplstyle')
def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']



#######################


### USER INPUT: ####

## Run input
run_list = ["SEXTRACTOR","STARSELECT","CREATEPSF"]
#run_list = ["STARSELECT","CREATEPSF"]

## Image input
image_pars = {"imgpath":"/Volumes/MyBook_18TB/data/Work/COSMOS/COSMOS-Web/data/images_Jan2023/v0.1/mosaic_nircam_f150w_COSMOS-Web_30mas_v0_1_i2d.fits",
                "hduext":"SCI",
                "lam":1.5 * u.micrometer
            }

## Paths
general_paths = {"output":"/Users/afaisst/Work/JWST_Projects/JWST_PSF_Pipeline/production/output/"}


## SExtractor input
sextractor_pars = {
    "call":"/Volumes/LaCie_2TB/data/Work/Tools/SExtractor/sextractor-2.19.5/bin/sex",
    "def_paths":{
                    "input":"../sextractor/",
                },
    "def_pars":{
                    "DETECT_THRESH":5,
                }
}


## Star selection
starselection_pars = {
    "mag_range_fit":[20,23], # magnitude range for fitting stellar locus
    "re_range_percent":[10,10], # radius range for star selection in percent from median radius
    "class_star_limit":0.8, # CLASS_STAR limit for star selection
    "makeplot":True # make figure
}

## Cutout and stacking
psf_pars = {
    "cutout_size_arcsec":1*u.arcsec,
    "mask_radius_arcsec":0.2*u.arcsec,
    "cut_pix": 2,
    "psf_name": None,
    "makeplot":True
}


## RUN:
results = jpp.jpp(image_pars , general_paths , sextractor_pars , starselection_pars , psf_pars , run_list)