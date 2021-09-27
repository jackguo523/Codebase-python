# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:59:18 2021

@author: Jack
"""

# virtual color conversion of MUSE imaging using the FalseColor module
# now support the convesion of eosin and Hoechst staining into H&E

import falsecolor.coloring as fc
# import numpy as np
import skimage.io
import skimage.color
import cv2
import glob
import tqdm
# import h5py as h5
import os

# read channels
# cell=skimage.io.imread('001.tif')[:,:,2]
# cyto=skimage.io.imread('001.tif')[:,:,1]
# file_path=os.path.join(os.getcwd(),'kidney_data.h5')

# with h5.File(file_path,'r') as f:
#     cell=f['t00000/s00/0/cells'][:]
#     cyto=f['t00000/s01/0/cells'][:]
# f.close()


inpath='test'
filemask=inpath+'/*.tif'
files=glob.glob(filemask)
outpath=inpath+'/result'
if not os.path.exists(outpath):
    os.mkdir(outpath)

for i in tqdm.tqdm(range(len(files)),desc='processing virtual mapping...'):   
    im=skimage.io.imread(files[i]) # read image
    tmp=skimage.color.rgb2lab(im) # convert from rgb to lab (this is tricky, found useful for tissue stained with eosin and DAPI)
    cell=tmp[:,:,2] # take the b-channel for nucleus
    cell=255-cv2.normalize(cell,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1) # normalize and invert
    cyto=tmp[:,:,0] # take the L-channel for cytoplasm
    cyto=cv2.normalize(cyto,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1) # normalize
    
    # subtract background
    cell_background=fc.getBackgroundLevels(cell)[1]
    cell=cell-cell_background
    # cell=np.clip(cell,0,65535)
    cyto_background=fc.getBackgroundLevels(cyto)[1]
    cyto=cyto-cyto_background
    # cyto=np.clip(cyto,0,65535)

    HE_settings=fc.getColorSettings(key='HE') # load default color settings
    result=fc.rapidFalseColor(cell,cyto,HE_settings['nuclei'],HE_settings['cyto'],nuc_normfactor=400,cyto_normfactor=400) # play with the numbers to determine the color space of the final image

    # save image
    skimage.io.imsave(outpath+'/{:03d}.tif'.format(i+1),result,check_contrast=False)
    # skimage.io.imsave('cell.tif',cell[0],check_contrast=False)
    # skimage.io.imsave('cyto.tif',cyto[0],check_contrast=False)