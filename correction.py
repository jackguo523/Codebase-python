# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:23:43 2020

@author: Jack
"""


# MUVE illumination provides uneven excitation on the tissue suraface that causes shadows and shades
# This script removes intensity shades using Gaussian convolution


import glob
import numpy as np
import skimage.io
import skimage.filters
import cv2 as cv
import time
import sys
import os
import scipy.signal
import scipy.ndimage
import skimage.morphology
import tqdm

root_filepath='tt'
file_format='*.tif'
outlier_removal=False
sigma=200
reference=True
idx=5 # reference frame for normalization

filemask=root_filepath+'/'+file_format   # create file mask
files=glob.glob(filemask)   # get all file names
numI=len(files)     # get the file count
I=[]    # input RGB images

t_start=time.time()

# read all images
for i in range(numI):
    # read all images from folder, note that cv2 uses the BGR format while skimage uses the RGB format
    I.append(skimage.io.imread(files[i]))
    #I.append(cv.imread(files[i]))

# get image dimensions
Ix=I[0].shape[1]   # get image width
Iy=I[0].shape[0]   # get image height

# corret and normalize all images
for i in tqdm.tqdm(range(numI),desc='background correction',position=0,leave=True): 
    tmpI=np.zeros([Iy,Ix,3],np.float32) # duplicate current image for gaussian blurring
    tmpI=np.float32(I[i])
    # kernel=np.outer(scipy.signal.gaussian(8*sigma+1,sigma),scipy.signal.gaussian(8*sigma+1,sigma))
    # tmpI[:,:,0]=scipy.signal.convolve(tmpI[:,:,0],kernel,mode='valid',method='fft')   
    # tmpI[:,:,1]=scipy.signal.convolve(tmpI[:,:,1],kernel,mode='valid',method='fft')   
    # tmpI[:,:,2]=scipy.signal.convolve(tmpI[:,:,2],kernel,mode='valid',method='fft')  
    # tmpI=skimage.filters.gaussian(tmpI,sigma,multichannel=True) # perform multichannel gaussian blurring
    # tmpI=cv.GaussianBlur(tmpI,(8*sigma+1,8*sigma+1),sigma,borderType=cv.BORDER_REPLICATE)
    # tmpI=scipy.ndimage.gaussian_filter(tmpI,sigma,mode='nearest')
    # tmpI=scipy.ndimage.uniform_filter(tmpI,2*sigma,mode='nearest') # tune the uniform filter to perform the same as the gaussian filter
    # tmpI=scipy.ndimage.gaussian_filter1d(tmpI,sigma,axis=0,mode='nearest')
    # tmpI=scipy.ndimage.gaussian_filter1d(tmpI,sigma,axis=1,mode='nearest')
    tmpI=scipy.ndimage.uniform_filter1d(tmpI,2*sigma,axis=0,mode='nearest')
    tmpI=scipy.ndimage.uniform_filter1d(tmpI,2*sigma,axis=1,mode='nearest')
    # skimage.io.imsave('2.tif',tmpI.astype(np.uint8))
    I[i]=I[i]/tmpI  # image division
    
    # perform outlier removal using median filter
    if outlier_removal:
        selem=skimage.morphology.disk(3)
        I[i][:,:,0]=skimage.filters.median(I[i][:,:,0],selem)
        I[i][:,:,1]=skimage.filters.median(I[i][:,:,1],selem)
        I[i][:,:,2]=skimage.filters.median(I[i][:,:,2],selem)
        
# create a folder for image saving
if not os.path.exists(root_filepath+'/T'):
        os.mkdir(root_filepath+'/T')
        
# perform normalization based upon the selected image as reference
rmax=I[idx][:,:,0].max()
gmax=I[idx][:,:,1].max()
bmax=I[idx][:,:,2].max()
rmin=I[idx][:,:,0].min()
gmin=I[idx][:,:,1].min()
bmin=I[idx][:,:,2].min()
for i in tqdm.tqdm(range(numI),desc='color normalization',position=0,leave=True):
    if reference: # normalize all images based on the selected reference image
        I[i][:,:,0]=np.round((I[i][:,:,0]-rmin)*((255-0)/(rmax-rmin))+0)
        I[i][:,:,1]=np.round((I[i][:,:,1]-gmin)*((255-0)/(gmax-gmin))+0)
        I[i][:,:,2]=np.round((I[i][:,:,2]-bmin)*((255-0)/(bmax-bmin))+0)
    else:   # normalize all images individually
        I[i][:,:,0]=cv.normalize(I[i][:,:,0],None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
        I[i][:,:,1]=cv.normalize(I[i][:,:,1],None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
        I[i][:,:,2]=cv.normalize(I[i][:,:,2],None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)
    
    skimage.io.imsave(root_filepath+'/T/'+'{:03d}.tif'.format(i+1),I[i].astype(np.uint8),check_contrast=False)
    #cv.imwrite(root_filepath+'/T/'+'{:03d}.tif'.format(i+1),np.uint8(I[i]))
 
t_end=time.time()
sys.stdout.write("\nTime to process: "+str(round(t_end-t_start,3))+"s\n")