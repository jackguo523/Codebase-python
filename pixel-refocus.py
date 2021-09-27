# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:58:27 2021

@author: Jack
"""

# this script does quick pixel-wise refocusing using normalized variance method


import os
import sys
import numpy as np
import time
import glob
from numba import cuda
import tqdm
import skimage.io
import cv2



@cuda.jit
def _copy(src,dst,ref):
    # tx,ty=cuda.grid(2)
    tx=cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x  # row
    ty=cuda.blockDim.y*cuda.blockIdx.y+cuda.threadIdx.y  # col
    
    if tx>=dst.shape[0] or ty>=dst.shape[1]:
        return  # avoid segmentation fault
    
    # if tx<dst.shape[0] and ty<dst.shape[1]:
    for d in range(dst.shape[2]):
        dst[tx,ty,d]=src[ref[tx,ty]][tx,ty,d]  # copy


@cuda.jit() 
def _rgb2gray(src,dst):
    tx=cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x  # row
    ty=cuda.blockDim.y*cuda.blockIdx.y+cuda.threadIdx.y  # col

    if tx>=dst.shape[0] or ty>=dst.shape[1]:
        return  # avoid segmentation fault
    
    dst[tx,ty]=round(0.2989*src[tx,ty,0]+0.5870*src[tx,ty,1]+0.1140*src[tx,ty,2])  # note that scikit-image uses [0.2125,0.7154,0.0721] coding


def stdout(words):
    sys.stdout.write(str(words)+'\n')
    sys.stdout.flush()
    

def load_stack_gpu(path,fformat='tif'):
    # (_,_,filenames) = next(os.walk(path))   # get all file names
    # fformat=filenames[0].split('.')[1]  # get the file suffix -- format
    # files=[path+'/'+s for s in filenames]   # get the file absolute directories
    filemask=path+'/*'+fformat  # get the file mask
    files=glob.glob(filemask)   # sort all files
    numI=len(files)     # get the file count
    I=[]    # RGB stack
    gI=[]   # grayscale stack
    
    for d in tqdm.tqdm(range(numI),desc='    loading image stack...',position=0,leave=True):
        # tmp=cv2.imread(files[d])  # opencv reads image as BGR whereas scikit reads images as RGB
        tmp=skimage.io.imread(files[d]) # read an image as is
        # tmp=Image.open(files[d])
        # tmp.draft('RGB',(270,512))  # load and shrink
        # tmp=np.asarray(tmp)
        
        if tmp.dtype==np.uint16:      # cast 16bit to 8bit and normalize
            tmp=cv2.normalize(tmp,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
        I.append(tmp)
        
        if tmp.ndim==3:  # create grayscale intensities
            sx=tmp.shape[1]  # get original dimensions
            sy=tmp.shape[0]
                        
            d_rgb=cuda.to_device(tmp)  # copy original rgb (source) from host to device
            d_gray=cuda.device_array((sy,sx),np.uint8)  # allocate memory for grayscale (destination) on device
            tmp=np.zeros((sy,sx),np.uint8)  # initialize grayscale intensities
            
            threadsperblock=(16,16)  # trial and error number -- maximum thread crushes
            blockspergrid_x=int(np.ceil(sy/threadsperblock[0]))
            blockspergrid_y=int(np.ceil(sx/threadsperblock[1]))
            blockspergrid=(blockspergrid_x,blockspergrid_y)
            
            _rgb2gray[blockspergrid,threadsperblock](d_rgb,d_gray)  # copy from original stack using gpu
            d_gray.copy_to_host(tmp)  # copy from device to host
        gI.append(tmp)
        
    return I,gI


def _var_map(img,win):
    import cupy  # import cupy
    from cupyx.scipy.ndimage import uniform_filter
    
    mean=uniform_filter(cupy.array(np.float32(img)),win,mode='reflect')  # get mean
    sqr_mean=uniform_filter(cupy.array(np.float32(img))**2,win,mode='reflect')  # get square mean
    tmp=sqr_mean-mean**2
    var=tmp.get()  # copy from device to host
    
    var=var/mean.get()  # normalization to compensate for the differences in average image intensity among different images

    return var


def refocus_gpu(rgbs,grays,win):
    V=[]    # initialize variance map
    numI=len(rgbs)
    
    for d in tqdm.tqdm(range(numI),desc='    computing variance map in parallel using gpu...',position=0,leave=True): # get variance map
        V.append(_var_map(grays[d],win))  # using cupy
    
    tmp=np.asarray(V)   # cast to an array for indexing
    H=np.uint8(tmp.argmax(axis=0))  # get height map by finding the largest variance position -- uint8 (255) might not be enough!!!
    # H=rip_bubble(H,3)
    
    sx=H.shape[1]   # get original dimensions
    sy=H.shape[0]
    ref=np.zeros((sy,sx,3),np.uint8)  # initialize refocus output
    
    stdout('    computing refocus fusion in parallel using gpu...')
    # device processing
    # gpu=cuda.get_current_device()  # get existing device
    # max_threadsperblock=gpu.MAX_THREADS_PER_BLOCK  # get maximum thread count
    d_rgbs=cuda.to_device(rgbs)  # copy original stack (source) from host to device
    d_H=cuda.to_device(H)  # copy height map (mask) from host to device
    d_ref=cuda.device_array((sy,sx,3),np.uint8)  # allocate memory for fusion (destination) on device
    
    threadsperblock=(16,16)  # trial and error number -- maximum thread crushes
    blockspergrid_x=int(np.ceil(sy/threadsperblock[0]))
    blockspergrid_y=int(np.ceil(sx/threadsperblock[1]))
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    
    _copy[blockspergrid,threadsperblock](d_rgbs,d_ref,d_H)  # copy from original stack using gpu parallel
    d_ref.copy_to_host(ref)  # copy from device to host
    
    H=numI-H-1  # note that the height map is inverted since the stage is moving from bottom to top. higher objects are first been imaged!!!
    
    return ref,V,H





rootpath='C:/Users/Jack/Desktop/10Xrefractive-15Xreflective/15Xreflective'  # root folder to process
folders=os.listdir(rootpath)  # detect all folders
folders = [i for i in folders if 'tif' not in i.lower()]  # delete previous saved tif files
block=51  # kernel size for convolution

t_start=time.time()     # start timer
for f in range(len(folders)):   # start batch process
    stdout('\nprocessing stack #'+str(f+1))
    fformat='.tif'     # file format
    tmp1,tmp2=load_stack_gpu(rootpath+'/'+folders[f],fformat)
    # result,_,hmap=refocus(tmp1,tmp2,block)
    result,_,hmap=refocus_gpu(tmp1,tmp2,block)
    skimage.io.imsave(rootpath+'/{:03d}'.format(f+1)+fformat,result,check_contrast=False)
    
t_end=time.time() # end timer
print('\n\ntime to process: '+str(round(t_end-t_start,3))+'s\n') 
    
    
    