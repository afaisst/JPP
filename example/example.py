import os,sys
import numpy as np
import time

from astropy.io import fits, ascii
from astropy.table import Table, hstack, vstack
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from astropy import units as u

import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("../code/")
import jpp as jpp

## Plotting stuff
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelpad'] = 10
mpl.rcParams['xtick.major.pad'] = 7
mpl.rcParams['ytick.major.pad'] = 7
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.minor.top'] = True
mpl.rcParams['xtick.minor.bottom'] = True
mpl.rcParams['ytick.minor.left'] = True
mpl.rcParams['ytick.minor.right'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
#mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['hatch.linewidth'] = 1

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
    "makeplot":True # make figure
}

## Cutout and stacking
cutout_pars = {
    "cutout_size_arcsec":1*u.arcsec,
    "mask_radius_arcsec":0.2*u.arcsec,
    "cut_pix": 2,
    "makeplot":True
}


## RUN:
results = jpp.jpp(image_pars , general_paths , sextractor_pars , starselection_pars , cutout_pars , run_list)