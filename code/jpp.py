### This Script should compute the PSF of a given JWST image automatically.
# Steps:
# 1. read in image
# 2. identify stars
# 3. extract stars
# 4. center and stack stars.


# TODO:
# - get lambda automatically
# - Add any SExtractor parameter to dictionary and SExtractor recognizes them.
# - Use PCA to get PSF variations across image

##### IMPORTS ########

import os, sys
import numpy as np
import time

from astropy.io import fits, ascii
from astropy.table import Table
from astropy.visualization import LogStretch, SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib as mpl


### DEFINITIONS #####

def get_diffraction_limit_psf_fwhm(lam):
    '''
    Computes the diffraction limited PSF FWHM for initial
    selection of stars on the MAG_AUTO vs. FLUX_RADIUS diagram

    Parameters:
    ----------
    - lam: `float`
        Wavelength of image in MICROMETERS

    Output:
    -------
    Returns the diffraction limited PSF FWHM in arcsec.

    '''
    D = 6.5 * u.meter
    psf_fwhm = 1.22 * lam.to(u.micrometer) / D.to(u.micrometer) * 180/np.pi * 3600

    return(psf_fwhm)


def getImageProperties(imgpath,hduext):
    '''
    Computes the some image header stuff for a given JWST image.
    This includes Zeropoint, HDU extension number, pixel scale, wcs
    Needs header keywords "PIXAR_A2" which is the pixscale**2 in arcsec^2. Also
    returns the HDU extension number for the `hduext' name and pixel scale.
    
    Parameters:
    -----------
    imgpath: `str`
        Path to the image
    hduext: `str`
        HDU extension (for example "SCI")
    
    Output:
    -------
    Returns zeropoint , hdu extension number , pixelscale , WCS
    '''
    
    # MJy/sr -> uJy/arcsec2: 23.5045
    # uJy/arcsec2 -> uJy: x pixscale**2 (e.g., 0.0009)
    # uJy -> mag has zeropoint of 23.9. Need to add that
    
    
    ## Load image
    with fits.open(imgpath, lazy_load_hdus=True) as hdul:
        hdr = hdul[hduext].header
        hdr_wcs = WCS(hdul[hduext].header)
        extlist = np.asarray([hdul[hh].header["EXTNAME"] for hh in range(1,len(hdul)) ])
        if hduext == "PRIMARY":
            hduextnbr = 0
        else:
            hduextnbr = np.where(hduext == extlist)[0][0] + 1 # because we cut the PRIMARY since it doesn't have an EXTNAME
                
    ## Compute zeropoint
    zp = -2.5*np.log10(23.5045 * hdr["PIXAR_A2"]) + 23.9
    
    ## Return
    return(zp , hduextnbr, np.sqrt(hdr["PIXAR_A2"]) , hdr_wcs)


def create_circular_mask(h, w, center=None, radius=None):
    '''
    Creates a cicular mask.

    Parameters:
    -----------
    h: `float`
        Height of image in pixels
    w: `float`
        With of image in pixels
    center: `tuple float`
        If center is not None, then it is used as center (h_center , w_center)
    radius: `float`
        Radius of circular mask. If None, it's the size of the image

    Output:
    --------
    Returns the mask as np.array()

    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def runsextractor(imgpath , hduextnbr, call, def_paths, def_pars):
    '''
    Runs Source Extractor on an image.

    Parameters:
    -----------
    - imgpath: `str`
        Path to image
    - hduextnbr: `int`
        Number of HDU extension (not name!)
    - call: `str`
        Location of Binary file to call SExtractor (typically 'sextractor')
    - def_paths: `dictionary`
        SExtractor default paths
            * input: input directory (typically 'sextractor/')
            * output: output directory where to store SExtractor catalog
    - def_pars: `dictionary`
        SExtractor parameters
            * DETECT_THRESH: detection threshold in sigmas
            * .... (more to be added soon)

    Output:
    -------
    Runs SExtractor on image and creates catalog. Returns path to catalog.

    TODO:
    -----
    - Add default parameter set for threshold, etc
    - Automate parameters, so it cycles through and adds the keys of the def_pars dictionary
    - Run in bins in parallel for large image files?

    '''

    ### Run SExtractor to extract sources. ####
    #dir_tmp = def_sextractor_paths["tmp"] # "../tmp/"
    #dir_sex = "../sextractor/"
    #dir_output = "../output/"

    sex_cat_output_name = imgpath.split("/")[-1].replace(".fits","_sex.dat")
    sex_cat_output_path = os.path.join(def_paths["output"] , sex_cat_output_name)


    cmd = "{} {}[{}] -c {} -CATALOG_NAME {}  -PARAMETERS_NAME {} -FILTER_NAME {} -STARNNW_NAME {}  -DETECT_THRESH {} -MAG_ZEROPOINT {} -PIXEL_SCALE {} -SEEING_FWHM {}".format(
        call,
        imgpath,
        hduextnbr,
        os.path.join(def_paths["input"] , "default.conf"),
        sex_cat_output_path,
        os.path.join(def_paths["input"] , "sex.par"),
        os.path.join(def_paths["input"] , "g2.8.conv"),
        os.path.join(def_paths["input"] , "default.nnw"),
        def_pars["DETECT_THRESH"],
        def_pars["MAG_ZEROPOINT"],
        def_pars["PIXEL_SCALE"],
        def_pars["SEEING_FWHM"]
    )
    print(cmd)

    results = os.system(cmd)

    print("Done!")
    return(
        {
            "cat_path":sex_cat_output_path
            }
    )

def select_stars_constant(cat , mag_range_fit , re_range_percent, lam, pixscale, output_path , MAKEPLOT):
    '''
    Selects stars from a SExtractor catalog according to a perfectly
    horizontal stellar locus. This is a VERY SIMPLE extraction method.
    Could try to do something more complicated. Note that the horizontal
    selection should get rid of saturated stars at the bright end automatically
    if they exist within the selected magnitude range.

    Parameters:
    ----------
    - cat: `astropy table`
        SExtractor catalog on which stars are selected. Must contain "MAG_AUTO" and "FLUX_RADIUS".
    - mag_range: `list, float`
        a list of bright and faint magnitude range to extract stars. Example: [20,26]
    - re_range_percent: `list, float`
        a list with the lower and upper size limit in percent. Example [20,20] for rmed +/- 0.2*rmet
    - lam: `float`
        Wavelength of image (used to compute diffraction limited PSF)
    - pixscale: `float`
        Pixel scale in arcsec/px
    - output_path: `str`
        Output path for plot (and catalog?)
    - MAKEPLOT: `bool`
        If set to True, creates a plot in the diagnostic directory.

    Output:
    --------
    - Returns a catalog of stars.
    - Makes a plot of MAG_AUTO vs. FLUX_RADIUS if MAKEPLOT = True

    Version:
    --------
    v1.0:
        - Initial version
    v1.1:
        - Improved selection of stars. Because JWST is diffraction limited, we can use that information
        to improve the star selection. The 2*re_range_percent is now around a diffraction limited PSF.
    '''

    ## Compute diffraction limited PSF FWHM. Note that JWST is basically
    # diffraction limited. See here:
    # - https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
    # - https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions
    # Note that in theory we also have to take into account the pixel size. However,
    # usually the pixel size is comparable or smaller than the JWST PSF size (=1).
    psf_fwhm_guess = get_diffraction_limit_psf_fwhm(lam = lam)
    fwhm_to_re = 0.5 # For a Gaussian, Re = FWHM*0.5 (for King profile it's 0.7)
    psf_re_total_pixel = np.sqrt( (psf_fwhm_guess/pixscale * fwhm_to_re)**2 + (1)**2 )
    print("RE SELECTION (px):" , psf_re_total_pixel.value)

    ## get data
    X = cat["MAG_AUTO"]
    Y = cat["FLUX_RADIUS"]

    ## Do initial selection. This includes candidate stars that are within
    # a range from a diffraction limited PSF.
    sel_stars1 =  np.where( (cat["CLASS_STAR"] > 0.9) 
                    & (cat["MAG_AUTO"] > mag_range_fit[0])
                    & (cat["MAG_AUTO"] < mag_range_fit[1])
                    & (cat["FLUX_RADIUS"] >= (psf_re_total_pixel.value - 2*re_range_percent[0]/100*psf_re_total_pixel.value))
                    & (cat["FLUX_RADIUS"] <= (psf_re_total_pixel.value + 2*re_range_percent[0]/100*psf_re_total_pixel.value))
                    )[0]
    Xstars1 = X[sel_stars1]
    Ystars1 = Y[sel_stars1]

    ## Compute median radius. Here we assume a fixed size as a
    # function of magnitude. This is very simple. Should get rid
    # of saturated sources naturally, if they don't dominate in the
    # selected magnitude range.
    re_med = np.median(Ystars1)

    ## Final selection:
    sel_stars2 = np.where( (cat["CLASS_STAR"] > 0.8) 
                    & (cat["MAG_AUTO"] > mag_range_fit[0])
                    & (cat["MAG_AUTO"] < mag_range_fit[1])
                    & (cat["FLUX_RADIUS"] >= (re_med - re_range_percent[0]/100*re_med))
                    & (cat["FLUX_RADIUS"] <= (re_med + re_range_percent[0]/100*re_med))
                    )[0]

    cat_stars = cat.copy()[sel_stars2]

    ## Save catalog
    cat_stars.write(os.path.join(output_path , "stars.dat") , format="ascii", overwrite=True)

    ## Make plot
    if MAKEPLOT:
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(1,1,1)

        X = cat["MAG_AUTO"]
        Y = cat["FLUX_RADIUS"]
        
        ax1.plot(X , Y , "o" , markersize=2 , color="lightgray" , alpha=0.5 , label="All detections")
        ax1.plot(X[sel_stars1] , Y[sel_stars1] , "o" , markersize=2 , color="gray" , alpha=0.8 , label="Candidate Stars")
        ax1.plot(X[sel_stars2] , Y[sel_stars2] , "x" , markersize=3 , color="red" , alpha=1 , label="Final Selection")
        ax1.axhline(y = psf_re_total_pixel , linewidth=1 , linestyle="--" , color="black")

        ax1.grid(linestyle=":",color="gray",linewidth=0.5)
        ax1.legend()
        ax1.set_yscale("log")
        ax1.set_xlabel("Magnitude (AB)")
        ax1.set_ylabel(r"$R_e$")

        plt.savefig(os.path.join(output_path , "starselection.pdf"),bbox_inches="tight")
        plt.close()

    return(cat_stars)


def RedefineRADEC(cat , img_wcs , output_path , origin):
    '''
    Recomputs the RA/DEC from the X_IMAGE and Y_IMAGE for a catalog.
    
    Parameters:
    ------------
    - cat: `astropy table`
        Catalog including X_IMAGE and Y_IMAGE
    - img_wcs: `astropy wcs object`
        WCS of image on which the catalog was obtained.
    - origin: `int`
        Origin of reference pixel (0 for Python, 1 for SExtractor)

    Output:
    --------
    New catalog with added "ALPHA_J2000_new" and "DELTA_J2000_new". Also saves the catalog again.
    Returns new catalog

    '''
    ## Get the correct RA/DEC from the image pixels ==========
    # Sometimes, SExtractor doesn't get it right. It is safer to compute
    # the coordinates again from the X_IMAGE and Y_IMAGE parameters.
    coords = np.zeros((len(cat),2))
    coords[:,0] = cat["X_IMAGE"]
    coords[:,1] = cat["Y_IMAGE"]
    coords_px = img_wcs.all_pix2world( coords , origin)
    cat["ALPHA_J2000_new"] = coords_px[:,0]
    cat["DELTA_J2000_new"] = coords_px[:,1]
    cat.write(os.path.join(output_path,"stars.dat") , format="ascii", overwrite=True)

    return(cat)


def StarsCutout(cat , imgpath, hduext , cutout_size_arcsec , ra_name , dec_name):
    '''
    Creates cutout of objects on a given image.

    Parameters:
    -----------
    - cat: `astropy table`
        Input catalog containing RA/DEC coordinates
    - imgpath: `str`
        Path to FITS image
    - hduext: `str`
        HDU extension name.
    - cutout_size_arcsec: `float`
        Cutout size in arcseconds
    - ra_name: `str`
        Column name for RA
    - dec_name: `str`
        Colum name for DEC

    Output:
    --------
    Returns a list of cutout in the same order as input catalog.

    '''
    

    STARS = []
    with fits.open(imgpath) as hdul:
        hdr = hdul[hduext].header
        this_wcs = WCS(hdul[hduext].header)

        pixscale = hdr["CDELT1"]*3600

        cutout_size_px = int( cutout_size_arcsec.to(u.arcsec).value / pixscale) * u.pixel

        if cutout_size_px.value % 2 == 0.0:
            cutout_size_px = cutout_size_px + 1*u.pixel
        
        for ii,star in enumerate(cat):
            
            #print("Cutting star ID={} at RA={}, DEC={}".format(star["NUMBER"],star["ALPHA_J2000"],star["DELTA_J2000"]))
            
            size = u.Quantity((cutout_size_px,cutout_size_px), u.pixel)
            position = SkyCoord(star[ra_name],star[dec_name] , unit="degree" , frame='icrs')
            cutout = Cutout2D(hdul[hduext].data, position, size, wcs=this_wcs , copy=True, mode="partial").data
            
            STARS.append(cutout)

    return(STARS)


def createPSF(stars , mask_radius_arcsec , pixscale , output_path , psf_name , cut_pix , MAKEPLOT):
    '''
    Creates the final PSF by centering and stacking the stars.

    Parameters:
    -----------
    - stars: `list np.array`
        List of stars (cutouts) as a 2x2 numpy array
    - mask_radius_arcsec: `int`
        Masking radius in arcseconds. This is used to center the stars.
    - pixscale: `float`
        Pixel scale of image.
    - output_path: `str`
        Path to output directory for saving PSF plot and FITS
    - psf_name: `str`
        Output name for this PSF
    - cut_pix: `int`
        Number of pixels to cut on the outside to remove dark frame introduced
        by shifting the individual cutouts to center. A value of 2 (pixels) is good.
    - MAKEPLOT: `bool`
        If set to true, a plot of the final PSF is created.

    Output:
    --------
    Creates a PSF and saves it in `output_path` location. Returns the final PSF as
    np.array object.

    '''

    ## First shift the stars to center them correctly in the 
    # middle of the cutout.
    STARS_SHIFT = []
    for star in stars:
        
        mask_radius = mask_radius_arcsec.value / pixscale # in pixels
        mask = create_circular_mask(h=star.shape[0],w=star.shape[1],radius=mask_radius)
        
        com = ndimage.center_of_mass(star*mask) # compute center of light in non-masked region
        shift = [ star.shape[0]//2 - com[0] , star.shape[1]//2 - com[1] ]
        star_shift = ndimage.shift(star, shift=shift )
        STARS_SHIFT.append( star_shift ) # shift
        
    ## Stack the shifted stars
    PSF = np.nanmedian( STARS_SHIFT , axis=0 ) # stack

    ## Cut the ugly edges introduced by shifting the images
    PSF = PSF[int(cut_pix):(PSF.shape[0]-int(cut_pix)+0)  , int(cut_pix):(PSF.shape[1]-int(cut_pix)+0) ]

    ## Normalize to 1
    PSF = PSF / np.nansum(PSF) # normalize

    ## Add some keywords to the header and write PSF to file.
    hdu0 = fits.PrimaryHDU(PSF)
    hdu0.header["NSTARS"] = int(len(stars))
    hdu0.header["PIXSCALE"] = round( pixscale ,3)
    hdu0.header["SIZE_AS"] = round( PSF.shape[0] * pixscale ,1)
    hdul = fits.HDUList([hdu0])
    hdul.writeto(os.path.join(output_path , psf_name) , overwrite=True)

    ## Make figure
    if MAKEPLOT:
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(1,1,1)

        ax1.imshow(PSF , origin="lower" , norm=ImageNormalize(stretch=LogStretch()))

        ax1.plot(PSF.shape[0]//2 , PSF.shape[1]//2 ,"x" , color="red")

        ax1.set_xlabel(r"x [px]")
        ax1.set_ylabel(r"y [px]")

        plt.savefig(os.path.join(output_path ,psf_name.replace(".fits",".pdf")),bbox_inches="tight")
        plt.close()


    return(PSF)



###########################


############# MAIN FUNCTION #################

def jpp(image_pars , general_paths , sextractor_pars , starselection_pars , psf_pars , run_list):
    '''
    Creates a PSF from a given image file.

    Parameters:
    -----------
    - image_pars: `dictionary`
        Image parameters, including:
            * imgpath: full path to the FITS image from which PSF should be created
            * hduext: Name of HDU extension. E.g., "PRIMARY" or "SCI" where the image data is located in FITS file
            * lam: Wavelength in MIRCOMETERS of the filter/image
    - general_paths: `dictionary`
        Storing all general paths:
            * output: Output main path (note that a sub-directory will be created for each image)
    - sextractor_pars: `dictionary`
        Dictionary containing parameters for running SExtractor:
            * call: Where the SExtractor binary sits (if not default 'sextractor')
            * def_paths: `dictionary`: Some paths:
                * input: where SExtractor input paramters are located (usually "../sextractor/")
            * def_pars: `dictionary`: some definition parameters
                * DETECT_THRESH: detection threshold
    - starselection_pars: `dictionary`
        Parameters for selection of stars
            * mag_range_fit: list of min/max magnitude for fitting stellar locus, e.g., [20,23]
            * re_range_percent: list of range for star selection in percent from medium size, e.g., [10,10]
            * makeplot: if True, creates a plot of selected stars on MAG_AUTO vs. FLUX_RADIUS diagram
    - psf_pars: `dictionary`
        Parameters for creating cutouts and stacking
            * cutout_size_arcsec: Angular size of cutout in ARCSECONDS
            * mask_radius_arcsec: Radius of mask applied to calculated the center of stars for centering in ARCSEC
            * cut_pix: Margin in PIXELS to cut off from final PSF (removes dark border due to shifting of stars)
            * psf_name: Set the name of the PSF (not path, just name). If "None", then the default PSF name is adopted
                        which is the image_name.replace(".fits","_psf.fits")
            * makeplot: if True, create plot of final PSF
    - run_list: `str list`
        List of commands to run. Full list: ["SEXTRACTOR","STARSELECT","CREATEPSF"]
    '''

    ## Time =====
    t1 = time.time()

    ## Talk ============
    print("Generating a PSF for image: {}".format( image_pars["imgpath"] ))

    ## Create New Directory ==========
    # Also update the paths
    general_paths["output_sub_path"] = image_pars["imgpath"].split("/")[-1].split(".fits")[0]
    general_paths["main_output"] = os.path.join( general_paths["output"] , general_paths["output_sub_path"] )
    sextractor_pars["def_paths"]["output"] = general_paths["main_output"]
    if os.path.exists( general_paths["main_output"] ):
        print("Directory {} exists.".format( general_paths["main_output"] ))
    else:
        cmd = "mkdir {}".format( general_paths["main_output"] )
        print(cmd)
        os.system(cmd)

    ## Get Zeropoint and HDU extension number ========
    zp , hduextnbr, pixscale , hdr_wcs = getImageProperties(image_pars["imgpath"] , image_pars["hduext"])
    image_pars["zp"] = zp
    image_pars["hduextnbr"] = hduextnbr
    image_pars["pixscale"] = pixscale
    image_pars["wcs"] = hdr_wcs
    sextractor_pars["def_pars"]["MAG_ZEROPOINT"] = zp
    sextractor_pars["def_pars"]["PIXEL_SCALE"] = pixscale
    sextractor_pars["def_pars"]["SEEING_FWHM"] = get_diffraction_limit_psf_fwhm(image_pars["lam"])

    ## First create SExtractor catalog =============
    if "SEXTRACTOR" in run_list:
        s_results = runsextractor(imgpath=image_pars["imgpath"],
                                hduextnbr=image_pars["hduextnbr"],
                                call=sextractor_pars["call"],
                                def_paths = sextractor_pars["def_paths"],
                                def_pars = sextractor_pars["def_pars"]
                                )
    else: # if SExtractor is not run, we have to create a fake dictionary containing the path to the catalog
        s_results = dict()
        s_results["cat_path"] = os.path.join( sextractor_pars["def_paths"]["output"] , image_pars["imgpath"].split("/")[-1].replace(".fits","_sex.dat") )
        


    ## Open catalog ===========
    scat = Table.read(s_results["cat_path"] , format="ascii.sextractor")
    print("{} sources in catalog!".format(len(scat)))


    ## Select Stars =============
    if "STARSELECT" in run_list:
        scat_stars = select_stars_constant(cat=scat, mag_range_fit=starselection_pars["mag_range_fit"],
            re_range_percent=starselection_pars["re_range_percent"] ,
            lam = image_pars["lam"],
            pixscale = image_pars["pixscale"],
            output_path = general_paths["main_output"],
            MAKEPLOT = starselection_pars["makeplot"]
        )
        print("{} Stars found!".format(len(scat_stars)))
    else:
        scat_stars = Table.read(os.path.join( general_paths["main_output"] , "stars.dat" ) , format="ascii")
        print("{} Stars found!".format(len(scat_stars)))

    ## Redefine RA and DEC from X_IMAGE and Y_IMAGE =======
    scat_stars = RedefineRADEC(cat=scat_stars,
        img_wcs = image_pars["wcs"],
        output_path = general_paths["main_output"],
        origin = 1
        )



    ## Create PSF ======================

    # PSF name
    if psf_pars["psf_name"] == None:
        general_paths["psf_name"] = image_pars["imgpath"].split("/")[-1].replace(".fits","_psf.fits")
    else:
        general_paths["psf_name"] = psf_pars["psf_name"]

    if "CREATEPSF" in run_list:

        ## Cut out stars ===============
        STARS = StarsCutout( cat = scat_stars,
                            imgpath = image_pars["imgpath"],
                            hduext = image_pars["hduext"],
                            cutout_size_arcsec = psf_pars["cutout_size_arcsec"],
                            ra_name = "ALPHA_J2000_new",
                            dec_name = "DELTA_J2000_new"
                            )

        ## Create PSF ================
        PSF = createPSF( stars = STARS,
                        mask_radius_arcsec = psf_pars["mask_radius_arcsec"],
                        pixscale = image_pars["pixscale"],
                        output_path = general_paths["main_output"],
                        psf_name = general_paths["psf_name"],
                        cut_pix = psf_pars["cut_pix"],
                        MAKEPLOT = psf_pars["makeplot"]
                        )

    t2 = time.time()
    print("All Done in {:.2f} minutes".format( (t2-t1)/60 ))
