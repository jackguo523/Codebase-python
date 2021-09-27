# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:50:05 2021

@author: Jack
"""

# this script maps color space to grayscale image

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import BoundaryNorm, ListedColormap
import glob
import cv2


def linear_colormap(gray,cmap='jet'):
    # gray=255-gray
    # gray=cv2.equalizeHist(gray)
    cm=plt.get_cmap(cmap)
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    
    result=cm(gray)  # apply colormap
    # skimage.io.imsave(output,(result[:,:,:3]*255).astype(np.uint8))
    
    return result

def get_mpl_colormap(cmap_name):
    cmap=plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm=plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range=sm.to_rgba(np.logspace(0, 1, 256),bytes=True)[:,2::-1]

    return color_range.reshape(256,1,3)

def nonlinear_colormap(gray,color,bound,output='result.tif'):
    cm=np.zeros((256,1,3),np.uint8)  # initialize color map
    
    for b in range(len(bound)):
        if b==0:
            low=0
        else:
            low=bound[b-1]+1
        up=bound[b]+1
    
        cm[low:up,0,0]=color[b][0]
        cm[low:up,0,1]=color[b][1]
        cm[low:up,0,2]=color[b][2]
    
    result=cv2.applyColorMap(gray,cm)
    skimage.io.imsave(output,result)
    
    return result



path='Shan/supplementary/raw'
out_dir='Shan/supplementary/cm1'
filemask=path+'/*.tif'
files=glob.glob(filemask)
ims=[]
imax=0
imin=255
for i in range(len(files)):
    tmp=skimage.io.imread(files[i])
    ims.append(tmp)
    if imax<tmp.max():
        imax=tmp.max()
    if imin>tmp.min():
        imin=tmp.min()
for i in range(len(ims)):
    tmp=np.uint8(np.round((ims[i]-imin)*((255-0)/(imax-imin))+0))
    # im=cv2.normalize(im,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
    result=linear_colormap(tmp)
    skimage.io.imsave(out_dir+'/{:03d}.tif'.format(i+1),(result[:,:,:3]*255).astype(np.uint8))
    
# RdYlBu
# bound=[1,10,30,50,80,90,100,150,200,225,255]
# color=[]
# ctp1=[165,0,38]
# color.append(ctp1)
# ctp2=[215,48,39]
# color.append(ctp2)
# ctp3=[244,109,67]
# color.append(ctp3)
# ctp4=[253,174,97]
# color.append(ctp4)
# ctp5=[254,224,144]
# color.append(ctp5)
# ctp6=[255,255,191]
# color.append(ctp6)
# ctp7=[224,243,248]
# color.append(ctp7)
# ctp8=[171,217,233]
# color.append(ctp8)
# ctp9=[116,173,209]
# color.append(ctp9)
# ctp10=[69,117,180]
# color.append(ctp10)
# ctp11=[49,54,149]
# color.append(ctp11)
# color.append(ctp7)

# A=get_mpl_colormap('RdYlBu')
# tmp=nonlinear_colormap(I,color,bound,'003.tif')