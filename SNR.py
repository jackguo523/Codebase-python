# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:13:49 2020

@author: Jack
"""

# This script computes both peak SNR and mean SNR based on an empty frame or blackfield


import glob
import numpy as np
import skimage.io
import skimage.filters
import os

# empty_frame='SNR/blackfield1.tif'
# sample_frame='SNR/sample.tif'

# # read images and convert them to grayscale
# black_img=skimage.io.imread(empty_frame)
# black_img=0.2989*black_img[:,:,0]+0.5870*black_img[:,:,1]+0.1140*black_img[:,:,2]
# black_img=np.uint8(np.round(black_img))
# img=skimage.io.imread(sample_frame)
# img=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
# img=np.uint8(np.round(img))

# noise_std=np.std(black_img)
# sample_max=np.max(img)
# sample_mean=np.mean(img)
# peak_snr=20*np.log10(sample_max/noise_std)
# mean_snr=20*np.log10(sample_mean/noise_std)

# brightfield='SNR/aligned_028.tif'
# darkfield='SNR/aligned_028-1.tif'
# img1=skimage.io.imread(brightfield)
# img2=skimage.io.imread(darkfield)
# img1=(skimage.color.rgb2gray(img1)*255).astype(np.uint8)
# img2=(skimage.color.rgb2gray(img2)*255).astype(np.uint8)
# img1=0.2989*img1[:,:,0]+0.5870*img1[:,:,1]+0.1140*img1[:,:,2]
# img1=np.uint8(np.round(img1))
# img2=0.2989*img2[:,:,0]+0.5870*img2[:,:,1]+0.1140*img2[:,:,2]
# img2=np.uint8(np.round(img2))


# noise=np.std(img2)
# img_max=np.max(img1)
# img_mean=np.mean(img1)
# peak_snr=20*np.log10(img_max/noise)
# mean_snr=20*np.log10(img_mean/noise)


def load_gray(path):
    tmpI=skimage.io.imread(path)
    result=skimage.color.rgb2gray(tmpI)*255
    # result=0.2989*tmpI[:,:,0]+0.5870*tmpI[:,:,1]+0.1140*tmpI[:,:,2]
    result=np.uint8(np.round(result))
    return result


filepath='SNR'
(_,folders,files)=next(os.walk(filepath))    # get the tile stacks, ground truth images, and log file
mean_snr_1=[]
mean_snr_2=[]
contrast_1=[]
contrast_2=[]
numF=len(folders)
for f in range(numF):
    filemask=filepath+'/'+folders[f]+'/*tif'
    I=glob.glob(filemask)
    img1=load_gray(I[0]) # muse
    img2=load_gray(I[1])
    img3=load_gray(I[2]) # Nikon
    img4=load_gray(I[3])

    noise_1=np.std(img2)
    noise_2=np.std(img4)
    mean_1=np.mean(img1)
    mean_2=np.mean(img3)
    
    mean_snr_1.append(20*np.log10(mean_1/noise_1))
    mean_snr_2.append(20*np.log10(mean_2/noise_2))
    
    max_1=np.max(img1)
    max_2=np.max(img3)
    min_1=np.min(img1)
    min_2=np.min(img3)
    
    contrast_1.append((max_1-min_1)/(max_1+min_1))
    contrast_2.append((max_2-min_2)/(max_2+min_2))
    
msnr_1=np.mean(mean_snr_1)
msnr_2=np.mean(mean_snr_2)
mcon_1=np.mean(contrast_1)
mcon_2=np.mean(contrast_2) 