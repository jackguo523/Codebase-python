# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:25:09 2021

@author: Jack
"""

# Thorcam by default saves 16bit RGB image, which is not supported by most WINDOW applications
# This light script converts a 16bit RGB image into a 8bit RGB image
# future might extend this to any bit depth

import cv2
import skimage
import glob
import os

in_dir='C:/Users/Jack/Desktop/data'
out_dir='C:/Users/Jack/Desktop/data/result'
fformat='.tif'

filemask=in_dir+'/*'+fformat
files=glob.glob(filemask)   # get all file names
numI=len(files)     # get the file count

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for i in range(numI):
    I=skimage.io.imread(files[i])
    I=cv2.normalize(I,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)  # cast to 8bit and normalize
    skimage.io.imsave(out_dir+'/'+'{:03d}.tif'.format(i+1),I)