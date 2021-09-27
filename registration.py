# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:18:53 2021

@author: JACK
"""

from pystackreg import StackReg
from image_registration import cross_correlation_shifts
import numpy as np
import skimage.io
import skimage.color
import skimage.transform
import scipy.ndimage
import cv2
import glob
import tqdm
import os


filemask = 'C:/Users/jack/Desktop/t/*.tif'     #### your input files - images
out_dir = 'C:/Users/jack/Desktop/t/aligned'	  #### output directory
# method='pystackreg'
method='image_registration'

files=glob.glob(filemask)   # get all file names
numI=len(files)     # get the file count

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

warps=np.eye(3,3,dtype=np.float64)  # initialize transformation matrix    
ref=skimage.io.imread(files[0])  # read the first frame
sx=ref.shape[1]  # get original dimensions
sy=ref.shape[0]
skimage.io.imsave(out_dir+'/aligned_001.tif',ref,check_contrast=False)
for d in tqdm.tqdm(range(1,numI),desc='starting image registration...',position=0,leave=True):
    ref=skimage.io.imread(files[d-1])  # read previous frame
    dec=skimage.io.imread(files[d])  # read a new frame
    
    ref_gray=skimage.color.rgb2gray(ref)  # convert from rgb to grayscale
    dec_gray=skimage.color.rgb2gray(dec)
    
    # ref_gray=scipy.ndimage.filters.gaussian_filter(ref_gray,(10,10))  # Gaussian blurring
    # dec_gray=scipy.ndimage.filters.gaussian_filter(dec_gray,(10,10))
    
    if method=='pystackreg':  # using pystackreg package
        sr=StackReg(StackReg.TRANSLATION)  # create registration object
        sr.register(ref_gray,dec_gray)  # find current transformation
        tmp=sr.get_matrix()  # get current transformation matrix
        warps[0,2]=warps[0,2]+tmp[0,2]  # accumulate previous transformation matrix
        warps[1,2]=warps[1,2]+tmp[1,2]
        sr.set_matrix(warps)  # set current transformation matrix
    elif method=='image_registration':  # using image_registration package
        xoff,yoff=cross_correlation_shifts(ref_gray,dec_gray)  # find current transformation
        warps[0,2]=warps[0,2]+xoff
        warps[1,2]=warps[1,2]+yoff
    
    r=cv2.warpAffine(dec[:,:,0],warps[0:2,:],(sx,sy),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)  # align individual channel
    g=cv2.warpAffine(dec[:,:,1],warps[0:2,:],(sx,sy),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    b=cv2.warpAffine(dec[:,:,2],warps[0:2,:],(sx,sy),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    # r=np.uint8(sr.transform(dec[:,:,0]))  # align individual channel
    # g=np.uint8(sr.transform(dec[:,:,1]))
    # b=np.uint8(sr.transform(dec[:,:,2]))
    result=np.dstack((r,g,b))  # merge aligned channels
    skimage.io.imsave(out_dir+'/aligned_'+'{:03d}.tif'.format(d+1),result,check_contrast=False)