# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:04:21 2021

@author: JACK
"""

# this script includes all features for processing MUSE-scanned images

# load image, compress image, refocus image, correct image, stitch image

# during acquisition, always from bottom to top (moving up!!!)

import os
import sys
import numpy as np
import argparse
import tqdm
import cv2
import glob
import time
from PIL import Image
import skimage.io
import skimage.filters
import skimage.feature
import scipy.ndimage
import joblib
from numba import cuda



# gpu kernels, note that x->row, y->col in cuda syntax
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
    
    # cuda.syncthreads()
    
@cuda.jit() 
def _rgb2gray(src,dst):
    tx=cuda.blockDim.x*cuda.blockIdx.x+cuda.threadIdx.x  # row
    ty=cuda.blockDim.y*cuda.blockIdx.y+cuda.threadIdx.y  # col

    if tx>=dst.shape[0] or ty>=dst.shape[1]:
        return  # avoid segmentation fault
    
    dst[tx,ty]=round(0.2989*src[tx,ty,0]+0.5870*src[tx,ty,1]+0.1140*src[tx,ty,2])  # note that scikit-image uses [0.2125,0.7154,0.0721] coding



# console print function
def stdout(words):
    sys.stdout.write(str(words)+'\n')
    sys.stdout.flush()


# rgb to grayscale function using rec601 luma coding
def __rgb2gray(rgb):
    tmp=0.2989*rgb[:,:,0]+0.5870*rgb[:,:,1]+0.1140*rgb[:,:,2]  # note that scikit-image uses [0.2125,0.7154,0.0721] coding
    gray=np.uint8(np.round(tmp))  # round it up
    
    return gray


# load image stack given absolute path and suffix (format)
# input: stack absolute path, file suffix or format
# output: RGB and grayscale stack in numpy.array
def load_stack(path,fformat='tif'):
    # (_,_,filenames) = next(os.walk(path))   # get all file names
    # fformat=filenames[0].split('.')[1]  # get the file suffix -- format
    # files=[path+'/'+s for s in filenames]   # get the file absolute directories
    filemask=path+'/*.'+fformat  # get the file mask
    files=glob.glob(filemask)   # sort all files
    numI=len(files)     # get the file count
    I=[]    # RGB stack
    gI=[]   # grayscale stack
    
    for d in tqdm.tqdm(range(numI),desc='    loading image stack...',position=0,leave=True):
        # tmp=cv2.imread(files[d])  # opencv reads image as BGR whereas scikit reads images as RGB
        tmp=skimage.io.imread(files[d]) # read an image as is
        if tmp.dtype==np.uint16:      # cast 16bit to 8bit and normalize
            tmp=cv2.normalize(tmp,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1)
        I.append(tmp)
        
        if tmp.ndim==3:  # create grayscale intensities
            tmp=__rgb2gray(tmp)
        gI.append(tmp)
        
    return I,gI


# load image stack in parallel using gpu given absolute path and suffix
# input: stack absolute path, file format
# output: RGB and grayscale stack in numpy.array
def load_stack_gpu(path,fformat='tif'):
    # (_,_,filenames) = next(os.walk(path))   # get all file names
    # fformat=filenames[0].split('.')[1]  # get the file suffix -- format
    # files=[path+'/'+s for s in filenames]   # get the file absolute directories
    filemask=path+'/*.'+fformat  # get the file mask
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
 

# this function should be combined with above function with an additional argument
# load image stack in parallel using cpu multi-threading given absolute path and suffix
# input: stack absolute path, file format
# output: RGB and grayscale stack in numpy.array
def load_stack_parallel(path,fformat='tif'):
    filemask=path+'/*.'+fformat  # get the file mask
    files=glob.glob(filemask)   # sort all files
    
    stdout('    loading image stack in parallel using cpu...')
    I=joblib.Parallel(n_jobs=-1)(  # -1 means using all CPUs
        joblib.delayed(skimage.io.imread)(f) for f in files  # read rgb stack in parallel using cpu
        )
    
    # gI=[]  # initialize grayscale stack
    if I[0].ndim==3:  # create grayscale intensities
        gI=joblib.Parallel(n_jobs=-1)(  # -1 means using all CPUs
            joblib.delayed(_rgb2gray)(i) for i in I  # convert rgb to grayscale
            )
    else:  # direct copy of intensities
        gI=I
    
    return I,gI


# contrast limited adaptive histogram equalization
# input: rgb image
# output: enhanced rgb image in numpy.array
def _clahe(rgb):  # define single clahe process
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)  # convert from rgb to hsv
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(100,100))  # create clahe object
    tmp=clahe.apply(hsv[:,:,2])  # apply clahe to brightness channel
    hsv[:,:,2]=np.uint8(tmp)  # copy brightness value
    result=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)  # convert from hsv to rgb
    
    return result


# outlier (ex. hot pixel) removal using size=2 median filter
# input: rgb image
# output: filtered rgb image in numpy.array
def outlier_removal(rgb):
    result=scipy.ndimage.median_filter(rgb,size=2,mode='mirror') # edges are handled using mirror
    
    return result


# contrast enhancement using contrast limited adaptive histogram equalization
# input: rgb images
# output: enhanced rgb images in numpy.array
def contrast_enhance(rgbs):
    numI=len(rgbs)  # get image count
    result=[]
    
    for d in tqdm.tqdm(range(numI),desc='    processing contrast enhancement...',position=0,leave=True):
        tmp=_clahe(rgbs[d])
        result.append(tmp)
    
    return result


# contrast enhancement in parallel using cpu multi-threading using contrast limited adaptive histogram equalization
# input: rgb images
# output: enhanced rgb images in numpy.array
def contrast_enhance_parallel(rgbs):
    stdout('    processing contrast enhancement in parallel using cpu...')
    
    result=joblib.Parallel(n_jobs=-1)(  # -1 means using all CPUs
        joblib.delayed(_clahe)(i) for i in rgbs  # read rgb stack in parallel using cpu
        )
    
    return result


# dilation of erosion -- morphological opening
# input: image, kernel size, kernel shape (ex. square, cross, circle)
# output: opened image
def morphology_opening(img,k=3,t='square'):
    if t=='square':     # square window
        kernel=np.ones((k,k),np.uint8)   # by default 5x5 kernel
    elif t=='cross':    # cross window
        kernel=np.zeros((k,k),np.uint8)
        mid=np.uint8(np.floor(k/2))     # get the center line
        kernel[mid,:]=1
        kernel[:,mid]=1
    elif t=='circle':   # circle window
        kernel=np.zeros((k,k),np.uint8)
        mid=np.uint8(np.floor(k/2))     # get the center line
        for i in range(k):
            for j in range(k):
                if np.abs((i-mid)**2+(j-mid)**2)<=mid**2:
                    kernel[i,j]=1
    else:   # consider as square window
        kernel=np.ones((k,k),np.uint8)
        
    # col,row=np.meshgrid(np.linspace(-1,1,k),np.linspace(-1,1,k))
    # dis=np.sqrt(col*col+row*row)    # get distance to center of the window
    # kernel=np.exp(-((dis-0.0)**2/(2.0*s**2)))
    tmp=cv2.erode(img,kernel,iterations=1)    # erode away boundary spikes
    result=cv2.dilate(tmp,kernel,iterations=1)   # connect diminished components
        
    return result


# erosion of dilation -- morphological closing
# input: image, kernel size, kernel shape (ex. square, cross, circle)
# output: closed image
def morphology_closing(img,k=3,t='square'):
    if t=='square':     # square window
        kernel=np.ones((k,k),np.uint8)   # by default 5x5 kernel
    elif t=='cross':    # cross window
        kernel=np.zeros((k,k),np.uint8)
        mid=np.uint8(np.floor(k/2))     # get the center line
        kernel[mid,:]=1
        kernel[:,mid]=1
    elif t=='circle':   # circle window
        kernel=np.zeros((k,k),np.uint8)
        mid=np.uint8(np.floor(k/2))     # get the center line
        for i in range(k):
            for j in range(k):
                if np.abs((i-mid)**2+(j-mid)**2)<=mid**2:
                    kernel[i,j]=1
    else:   # consider as square window
        kernel=np.ones((k,k),np.uint8)
        
    # col,row=np.meshgrid(np.linspace(-1,1,k),np.linspace(-1,1,k))
    # dis=np.sqrt(col*col+row*row)    # get distance to center of the window
    # kernel=np.exp(-((dis-0.0)**2/(2.0*s**2)))
    tmp=cv2.dilate(img,kernel,iterations=1)    # connect diminished components
    result=cv2.erode(tmp,kernel,iterations=1)   # erode away boundary spikes
        
    return result


# image denoising using different methods: median filtering, Gaussian filtering, bilateral filtering, non-local means, wavelet
# input: noisy image, denoising method, corresponding values (e.g. kernel sigma for Gaussian)
# output: denoised image in numpy.array
def denoise(img,method='median',value=3):
    if method=='median':  # using median filtering
        result=scipy.ndimage.median_filter(img,size=value,mode='reflect')
    elif method=='gaussian':  # using Gaussian filtering
        R=scipy.ndimage.gaussian_filter(img[:,:,0],sigma=value,mode='reflect')
        G=scipy.ndimage.gaussian_filter(img[:,:,1],sigma=value,mode='reflect')
        B=scipy.ndimage.gaussian_filter(img[:,:,2],sigma=value,mode='reflect')
        result=np.dstack((R,G,B))
    elif method=='bilateral':  # using Bilateral filtering
        result=cv2.bilateralFilter(img,5,20,100,borderType=cv2.BORDER_CONSTANT)
    elif method=='nlm':  # using non-local means
        result=skimage.restoration.denoise_nl_means(img,patch_size=value,multichannel=True)
    elif method=='wavelet':
        result=skimage.restoration.denoise_wavelet(img,sigma=None,multichannel=True)
        
    return result
    

# blend two images with fixed alpha rate [average blending]
# input: 1st image, 2nd image, final image width, final image height, mosaic shift in (x,y), fixed alpha rate
# output: blend image in PIL.Image
def average_blend_two(img1,img2,width,height,shift,a=0.5):
    tmp1=Image.new('RGBA',size=(width,height),color=(0,0,0,0))
    tmp1.paste(img2,shift)   # paste tmp1 on top of tmp2
    tmp1.paste(img1,(0,0))
    
    tmp2=Image.new('RGBA',size=(width,height),color=(0,0,0,0))
    tmp2.paste(img1,(0,0))   # paste tmp2 on top of tmp1
    tmp2.paste(img2,shift)
    
    b=Image.blend(tmp1,tmp2,alpha=a)   # equally blend
    b=b.convert('RGB')  # convert to RGB
    
    return b


# blend two images with changing alpha rate [linear blending]
# input: 1st image, 2nd image, final image width, final image height, mosaic shift in (x or y) and overlaid pixel count depending on the axis (axis=0->x, axis=1->y)
# output: blend image in PIL.Image
def linear_blend_two(img1,img2,width,height,shift,overlap,axis=0):
    Image.MAX_IMAGE_PIXELS=width*height  # specify memory limit to prevent DecompressionBombError
    b=Image.new('RGBA',size=(width,height),color=(0,0,0,0))  # initialize blend
    
    if axis==0:  # horizontal blend
        tmp1=img1.crop((0,0,shift,height))
        b.paste(tmp1,(0,0))  # paste the non-shared part of img1
        tmp2=img2.crop((overlap,0,img2.size[0],height))
        b.paste(tmp2,(img1.size[0],0))  # paste the non-shaed part of img2
        
        tmp1=img1.crop((shift,0,img1.size[0],height))  # get the shared part of img1
        tmp2=img2.crop((0,0,overlap,height))  # get the shared part of img2
        
        mask=np.repeat(np.tile(np.linspace(1,0,overlap),(tmp1.size[1],1))[:,:,np.newaxis],3,axis=2)   # create gradient mask
        tmp=np.uint8(np.array(tmp1)*mask+np.array(tmp2)*(1-mask))  # linear blend
        tmp=Image.fromarray(tmp)
        
        b.paste(tmp,(shift,0))  # paste the blend
        b=b.convert('RGB')  # convert to RGB
    elif axis==1:  # vertical blend
        tmp1=img1.crop((0,0,width,shift))
        b.paste(tmp1,(0,0))  # paste the non-shared part of img1
        tmp2=img2.crop((0,overlap,width,img2.size[1]))
        b.paste(tmp2,(0,img1.size[1]))  # paste the non-shaed part of img2
        
        tmp1=img1.crop((0,shift,width,img1.size[1]))  # get the shared part of img1
        tmp2=img2.crop((0,0,width,overlap))  # get the shared part of img2
        
        mask=np.repeat(np.transpose(np.tile(np.linspace(1,0,overlap),(tmp1.size[0],1)))[:,:,np.newaxis],3,axis=2)   # create gradient mask
        tmp=np.uint8(np.array(tmp1)*mask+np.array(tmp2)*(1-mask))  # linear blend
        tmp=Image.fromarray(tmp)
        
        b.paste(tmp,(0,shift))  # paste the blend
        b=b.convert('RGB')  # convert to RGB
    
    return b


# currently only support images of the same size and large overlap rate > 15%
# register two images using cross-correlation
# input: reference image, offset image, offset direction img2->img1
# output: rigid shift
def correlation_register_two(img1,img2,direction='right'):
    img1=skimage.color.rgb2gray(img1)  # convert to grayscale
    img2=skimage.color.rgb2gray(img2)
    sx=img1.shape[1]  # get original dimensions
    sy=img1.shape[0]
    
    if direction=='right':  # left img1, right img2
        img1=img1[:,int(sx/2):sx]  # crop right
        img2=img2[:,0:int(sx/2)]  # crop left
        shift,_,_=skimage.feature.register_translation(img1,img2)
        shift[1]=sx+shift[1]
    elif direction=='left':  # left img2, right img1
        img1=img1[:,0:int(sx/2)]  # crop left
        img2=img2[:,int(sx/2):sx]  # crop right
        shift,_,_=skimage.feature.register_translation(img1,img2)
        shift[1]=sx-shift[1]
    elif direction=='bottom':  # top img1, bottom img2
        img1=img1[int(sy/2):sy,:]  # crop bottom
        img2=img2[0:int(sy/2),:]  # crop top
        shift,_,_=skimage.feature.register_translation(img1,img2)
        shift[0]=sy+shift[0]
    elif direction=='top':  # top img2, bottom img1
        img1=img1[0:int(sy/2),:]  # crop top
        img2=img2[int(sy/2):sy,:]  # crop bottom
        shift,_,_=skimage.feature.register_translation(img1,img2)
        shift[0]=sy-shift[0]
        
    return shift


# note this method losses information in cross regions
# stitch snake images using blending: average, linear
# input: mosaic images, tile row count, tile column count, overlap rate, blend mode
# output: stithced mosaic in PIL.Image
# in PIL.Image [0]->width, [1]->height, in numpy.array [0]->height, [1]->width
def stitch_snake(imgs,row,col,overlap,blend='linear'): 
    xshift=np.floor(imgs[0].shape[1]*(100-overlap)/100.0).astype(np.uint16)   # get single column shift
    yshift=np.floor(imgs[0].shape[0]*(100-overlap)/100.0).astype(np.uint16)   # get single row shift
    xoverlap=imgs[0].shape[1]-xshift
    yoverlap=imgs[0].shape[0]-yshift
    
    row_tiles=[]    # first row stitching
    for i in tqdm.tqdm(range(row),desc='    starting row stitching...',position=0,leave=True):
        rs=Image.fromarray(imgs[i*col])  # get the first image in current row
        for j in range(1,col):
            ow,oh=rs.size  # get original dimensions
            nw=ow+xshift  # get new width
            nh=oh  # get new height
            tmp=Image.fromarray(imgs[i*col+j])  # get a new image
            if i%2==0:  # moving right in odd rows
                if blend=='average':
                    rs=average_blend_two(rs,tmp,nw,nh,(j*xshift,0))
                elif blend=='linear':
                    rs=linear_blend_two(rs,tmp,nw,nh,j*xshift,xoverlap,0)
            else:   # moving left in even rows
                if blend=='average':
                    rs=average_blend_two(tmp,rs,nw,nh,(xshift,0))
                elif blend=='linear':
                    rs=linear_blend_two(tmp,rs,nw,nh,xshift,xoverlap,0)
        row_tiles.append(rs)
    
    cs=row_tiles[0] # then column stitching
    for d in tqdm.tqdm(range(1,row),desc='    starting column stitching...',position=0,leave=True):
        ow,oh=cs.size   # get original dimensions
        nw=ow   # get new width
        nh=oh+yshift  # get new height
        tmp=row_tiles[d]  # get a new image
        if blend=='average':
            cs=average_blend_two(cs,tmp,nw,nh,(0,d*yshift))
        elif blend=='linear':
            cs=linear_blend_two(cs,tmp,nw,nh,d*yshift,yoverlap,1)
        
    return cs


# general image stitch function
# input: mosaic images, tile row count, tile column count, overlap rate, scan mode [snake, raster]
# output: stitched mosaic in numpy.array
def stitch(imgs,row,col,overlap,scan='snake'):
    numI=len(imgs)
    
    # calculate final image size
    sx=imgs[0].shape[1]  # get original dimensions
    sy=imgs[0].shape[0]
    xshift=np.floor(sx*(100-overlap)/100.0).astype(np.uint16)  # get single column shift
    yshift=np.floor(sy*(100-overlap)/100.0).astype(np.uint16)  # get single row shift
    xoverlap=sx-xshift  # get column overlap
    yoverlap=sy-yshift  # get row overlap
    
    xfinal=col*xshift+xoverlap  # get final fusion size
    yfinal=row*yshift+yoverlap
    
    # s=Image.new('RGBA',size=(xfinal,yfinal),color=(0,0,0,0))  # initialize final stitch
    if imgs[0].ndim==3:  # rgb
        s=np.zeros((yfinal,xfinal,3),np.float64)  # initialize final stitch in float64 for blending
    else:  # grayscale
        s=np.zeros((yfinal,xfinal),np.float64)
    M=[]  # initialize individual mask

    if scan=='snake':  
        for i in tqdm.tqdm(range(numI),desc='    creating stitch mask...',position=0,leave=True):  # create stitch mask
            mask=np.ones((sy,sx,3),np.float64)  # initialize individual mask
            iy=int(i/col)  # get coordinates in 2d grid
            ix=(col-1-i%col,i%col)[iy%2==0]
    
            if iy!=0:  # except first row
                tmp=np.repeat(np.transpose(np.tile(np.linspace(1,0,yoverlap),(sx,1)))[:,:,np.newaxis],3,axis=2)  # create gradient mask
                mask[:yoverlap,:,:]=mask[:yoverlap,:,:]*(1-tmp)
            if iy!=(row-1):  # except last row
                tmp=np.repeat(np.transpose(np.tile(np.linspace(1,0,yoverlap),(sx,1)))[:,:,np.newaxis],3,axis=2)  # create gradient mask
                mask[-yoverlap:,:,:]=mask[-yoverlap:,:,:]*tmp
            if ix!=0: # except first column
                tmp=np.repeat(np.tile(np.linspace(1,0,xoverlap),(sy,1))[:,:,np.newaxis],3,axis=2)  # create gradient mask
                mask[:,:xoverlap,:]=mask[:,:xoverlap,:]*(1-tmp)
            if ix!=(col-1):  # except last column
                tmp=np.repeat(np.tile(np.linspace(1,0,xoverlap),(sy,1))[:,:,np.newaxis],3,axis=2)  # create gradient mask
                mask[:,-xoverlap:,:]=mask[:,-xoverlap:,:]*tmp
                
            M.append(mask)
    
        for i in tqdm.tqdm(range(numI),desc='    creating refocus fusion...',position=0,leave=True):  # create final refocus stitch
            iy=int(i/col)  # get coordinates in 2d grid
            ix=(col-1-i%col,i%col)[iy%2==0]
            
            oy=iy*yshift  # translate into actual position
            ox=ix*xshift
            
            if imgs[0].ndim==3:  # rgb
                tmp=imgs[i]*M[i]  # update image with computed mask
            else:  # grayscale
                tmp=imgs[i]*M[i][:,:,0]
                
            s[oy:oy+sy,ox:ox+sx]=s[oy:oy+sy,ox:ox+sx]+tmp  # update final fusion
            # tmp=Image.fromarray(tmp)
            # s.paste(tmp,(ox,oy))  # paste updated image
    elif scan=='raster':
        for i in tqdm.tqdm(range(numI),desc='    creating stitch mask...',position=0,leave=True):  # create stitch mask
            mask=np.ones((sy,sx,3),np.float64)  # initialize individual mask
            iy=int(i/col)  # get coordinates in 2d grid
            ix=i%col
    
            if iy!=0:  # except first row
                mask[:yoverlap,:,:]=mask[:yoverlap,:,:]*0.5
            if iy!=(row-1):  # except last row
                mask[-yoverlap:,:,:]=mask[-yoverlap:,:,:]*0.5
            if ix!=0: # except first column
                mask[:,:xoverlap,:]=mask[:,:xoverlap,:]*0.5
            if ix!=(col-1):  # except last column
                mask[:,-xoverlap:,:]=mask[:,-xoverlap:,:]*0.5
                
            M.append(mask)
    
        for i in tqdm.tqdm(range(numI),desc='    creating refocus fusion...',position=0,leave=True):  # create final refocus stitch
            iy=int(i/col)  # get coordinates in 2d grid
            ix=i%col
            
            oy=iy*yshift  # translate into actual position
            ox=ix*xshift
            
            if imgs[0].ndim==3:  # rgb
                tmp=imgs[i]*M[i]  # update image with computed mask
            else:  # grayscale
                tmp=imgs[i]*M[i][:,:,0]
            
            s[oy:oy+sy,ox:ox+sx]=s[oy:oy+sy,ox:ox+sx]+tmp  # update final fusion
            # tmp=Image.fromarray(tmp)
            # s.paste(tmp,(ox,oy))  # paste updated image
    
    # s=np.uint8(s)  # convert to unit8
    s=Image.fromarray(np.uint8(s))  # convert to uint8 PIL.Image
    
    return s


# variance map using variance trick: sum of square - square of sum (using scipy)
# input: process image, window size
# output: variance map
def var_map(img,win):
    mean=scipy.ndimage.uniform_filter(np.float32(img),win,mode='reflect')  # get mean
    sqr_mean=scipy.ndimage.uniform_filter(np.float32(img)**2,win,mode='reflect')  # get square mean
    var=sqr_mean-mean**2
    
    var=var/mean  # normalization to compensate for the differences in average image intensity among different images
    
    return var


# variance map using variance trick: sum of square - square of sum (parallel gpu processing using cupy)
# input: process image, window size
# output: variance map
def _var_map(img,win):
    import cupy  # import cupy
    from cupyx.scipy.ndimage import uniform_filter
    
    mean=uniform_filter(cupy.array(np.float32(img)),win,mode='reflect')  # get mean
    sqr_mean=uniform_filter(cupy.array(np.float32(img))**2,win,mode='reflect')  # get square mean
    tmp=sqr_mean-mean**2
    var=tmp.get()  # copy from device to host
    
    var=var/mean.get()  # normalization to compensate for the differences in average image intensity among different images

    return var


# remove refocus artifacts
# input: height map, filter window size
# output: corrected height map
def rip_bubble(hmap,win):
    selem=np.ones((win,win),bool)
    tmp=np.zeros((hmap.shape[0],hmap.shape[1]),np.uint8)
    skimage.filters.rank.majority(hmap,selem,out=tmp)  # using majority filter
    
    return tmp

            
# refocus tile -- could also do the rgb2gray conversion here
# input: RGB stack, grayscale stack, window size
# output: refocused image in numpy.array
def refocus(rgbs,grays,win):
    V=[]    # initialize variance map
    numI=len(rgbs)
    
    for d in tqdm.tqdm(range(numI),desc='    computing variance map...',position=0,leave=True): # get variance map
        V.append(var_map(grays[d],win))  # using scipy
    
    tmp=np.asarray(V)   # cast to an array for indexing
    H=np.uint8(tmp.argmax(axis=0))  # get height map by finding the largest variance position -- uint8 (255) might not be enough!!!
    # H=rip_bubble(H,3)
    
    sx=H.shape[1]   # get original dimensions
    sy=H.shape[0]
    ref=np.zeros((sy,sx,3),np.uint8)  # initialize refocus output
    
    for i in tqdm.tqdm(range(sy),desc='    computing refocus fusion...',position=0,leave=True):
        for j in range(sx):
            ref[i,j,:]=rgbs[H[i,j]][i,j,:]  # copy from original stack using BF method
    
    H=numI-H-1  # note that the height map is inverted since the stage is moving from bottom to top. higher objects are first been imaged!!!
    
    return ref,V,H


# refocus tile in parallel using gpu
# input: RGB stack, grayscale stack, window size
# output: refocused image in numpy.array
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
    
    # H=morphology_opening(H,201,'circle')
    
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
    

# refocus tile in parallel using cpu multi-threading -- *****currently not working*****
# input: RGB stack, grayscale stack, window size
# output: refocused tile in numpy.array
def refocus_parallel(rgbs,grays,win):
    stdout('    computing variance map in parallel using cpu...')
    V=joblib.Parallel(n_jobs=-1)(
        joblib.delayed(var_map)(i,win) for i in grays  # get variance map in parallel using cpu
        )
    
    tmp=np.asarray(V)   # cast to an array for indexing
    H=np.uint8(tmp.argmax(axis=0))  # get height map by finding the largest variance position -- uint8 (255) might not be enough!!!

    def _assign(x,y,S,M,i,j):  # _assign function
        result=np.zeros((y,x,3),np.uint8)  # initialize refocus output
        result[i,j,:]=S[M[i,j]][i,j,:]
        return result
    
    sx=H.shape[1]   # get original dimensions
    sy=H.shape[0]
    ref=np.zeros((sy,sx,3),np.uint8)  # initialize refocus output
    for i in tqdm.tqdm(range(sy),desc='    computing refocus fusion...',position=0,leave=True):
        for j in range(sx):
            ref[i,j,:]=rgbs[H[i,j]][i,j,:]  # copy from original stack
            
    H=len(rgbs)-H-1  # note that the height map is inverted since the stage is moving from bottom to top. higher objects are first been imaged!!!
    
    return ref,V,H


# intra-plane methods inevitably influence color information
# shading correction and normalization using Gaussian blurring
# input: corrupted images, convolution sigma, reference image index
# output: corrected images in numpy.array
def shade_correct_separate(imgs,sigma=300,index=0):
    numI=len(imgs)  # get image count
    sx=imgs[0].shape[1] # get dimensions
    sy=imgs[0].shape[0]
    result=[]
    
    for d in tqdm.tqdm(range(numI),desc='    starting shading correction...',position=0,leave=True):  # shading correction via Gaussian blurring
        tmp=np.zeros([sy,sx,3],np.float32)    
        tmp=np.float32(imgs[d])  # duplication in 32bit
        tmp=scipy.ndimage.uniform_filter1d(tmp,2*sigma,axis=0,mode='reflect')  # two times of blurring
        tmp=scipy.ndimage.uniform_filter1d(tmp,2*sigma,axis=1,mode='reflect')
        result.append(imgs[d]-tmp)  # image division or image subtraction
        
    ### need to think about a way to normalize
    # rmax=result[index][:,:,0].max()
    # gmax=result[index][:,:,1].max()
    # bmax=result[index][:,:,2].max()
    # rmin=result[index][:,:,0].min()
    # gmin=result[index][:,:,1].min()
    # bmin=result[index][:,:,2].min()
    for d in tqdm.tqdm(range(numI),desc='    starting color normalization',position=0,leave=True):  # color normalization
        # result[d]=cv2.normalize(result[d],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        result[d][:,:,0]=cv2.normalize(result[d][:,:,0],None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1) # R
        skimage.io.imsave('r'+str(d)+'.tif',result[d][:,:,0])
        result[d][:,:,1]=cv2.normalize(result[d][:,:,1],None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1) # G
        skimage.io.imsave('g'+str(d)+'.tif',result[d][:,:,1])
        result[d][:,:,2]=cv2.normalize(result[d][:,:,2],None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1) # B
        skimage.io.imsave('b'+str(d)+'.tif',result[d][:,:,2])
        # result[d][:,:,0]=np.round((result[d][:,:,0]-rmin)*((255-0)/(rmax-rmin))+0)
        # result[d][:,:,1]=np.round((result[d][:,:,1]-gmin)*((255-0)/(gmax-gmin))+0)
        # result[d][:,:,2]=np.round((result[d][:,:,2]-bmin)*((255-0)/(bmax-bmin))+0)

    return result


# this is good only for small images
# shading correction using rolling ball background subtraction from ImageJ
# input: corrupted images, rolling ball radius
# output: corrected images in numpy.array
def shade_correct_rolling_ball(imgs,radius=300):
    numI=len(imgs)  # get image count
    tmp=np.zeros((imgs[0].shape[0],imgs[0].shape[1],3),np.uint8)
    result=[]
    
    import cv2_rolling_ball
    for d in tqdm.tqdm(range(numI),desc='    starting shading correction...',position=0,leave=True):
        tmp[:,:,0],_=cv2_rolling_ball.subtract_background_rolling_ball(imgs[d][:,:,0],radius)
        tmp[:,:,1],_=cv2_rolling_ball.subtract_background_rolling_ball(imgs[d][:,:,1],radius)
        tmp[:,:,2],_=cv2_rolling_ball.subtract_background_rolling_ball(imgs[d][:,:,2],radius)
        result.append(tmp)

    return result


# intensity-based shading correction using hsv color space in spatial domain
# input: corrupted images, convolution sigma, reference image index
# output: corrected images in numpy.array
def shade_correct_intensity(imgs,sigma=300,index=0):
    numI=len(imgs)  # get image count
    inter_result=[]
    result=[]
    drange=[0,255]  # dynamic range
    
    for d in tqdm.tqdm(range(numI),desc='    starting shading correction...',position=0,leave=True):
        hsv=cv2.cvtColor(imgs[d],cv2.COLOR_RGB2HSV)  # convert from rgb to hsv
        v=np.zeros([hsv.shape[0],hsv.shape[1]],hsv.dtype)
        v=hsv[:,:,2]  # get brightness channel
        
        if d==index:
            drange[1]=v.max()  # get original brightness max of reference
            drange[0]=v.min()  # get original brightness min of reference
        tmp=np.zeros([v.shape[0],v.shape[1]],np.float32)
        tmp=np.float32(v)  # duplicate brightness channel as float32
        tmp=scipy.ndimage.uniform_filter1d(tmp,2*sigma,axis=0,mode='reflect') # two times of blurring
        tmp=scipy.ndimage.uniform_filter1d(tmp,2*sigma,axis=1,mode='reflect')
        # unsharp masking uses subtraction
        v=v-tmp  # image division or image subtraction
        # v=cv2.normalize(v,None,0,255,cv2.NORM_MINMAX,cv2.CV_32F)  # normalize intensity individually
        
        tmp=np.float32(hsv)  # copy original hsv as float32
        tmp[:,:,2]=v  # paste corrected value
        inter_result.append(tmp)
        # hsv[:,:,2]=np.uint8(np.floor(v))  # paste corrected value
        # rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)  # convert from hsv to rgb
        # result.append(rgb)
        
    vmax=inter_result[index][:,:,2].max()  # get current brightness max of reference
    vmin=inter_result[index][:,:,2].min()  # get current brightness min of reference
    for d in tqdm.tqdm(range(numI),desc='    starting intensity normalization...',position=0,leave=True):
        inter_result[d][:,:,2]=(inter_result[d][:,:,2]-vmin)*((drange[1]-drange[0])/(vmax-vmin))+drange[0]  # normalize all frames based on current reference to original reference
        tmp=np.maximum(0.0,np.minimum(inter_result[d],255.0))  # confine data to [0,255]
        tmp=np.uint8(np.floor(tmp))  # convert to uint8
        result.append(cv2.cvtColor(tmp,cv2.COLOR_HSV2RGB))  # convert from hsv to rgb

    # for d in tqdm.tqdm(range(numI),desc='    starting intensity normalization...',position=0,leave=True):
    #     result[d]=skimage.exposure.match_histograms(result[d],result[index],multichannel=True)  # histogram matching

    return result


# intensity-based shading correction using hsv color space in frequency domain
# input: corrupted images, lowpass size, reference image index
# output: corrected images in numpy.array
def shade_correct_intensity_lowpass(imgs,sigma=30,index=0):
    numI=len(imgs)  # get image count
    sx=imgs[0].shape[1]  # get original dimensions
    sy=imgs[0].shape[0]
    inter_result=[]
    result=[]
    drange=[0,255]  # dynamic range
    
    low_pass=Image.new('L',size=(sx,sy),color=0)  # initialize low-pass filter
    from PIL import ImageDraw
    tmp=ImageDraw.Draw(low_pass)  # draw a low-pass filter on an image
    bbox=((sx/2)-(sigma/2),(sy/2)-(sigma/2),(sx/2)+(sigma/2),(sy/2)+(sigma/2))
    tmp.ellipse(bbox,fill=1)  # circle in the center
    low_pass=np.array(low_pass)
    
    from scipy import fftpack
    for d in tqdm.tqdm(range(numI),desc='    starting shading correction...',position=0,leave=True):
        hsv=cv2.cvtColor(imgs[d],cv2.COLOR_RGB2HSV)  # convert from rgb to hsv
        v=np.zeros([hsv.shape[0],hsv.shape[1]],hsv.dtype)
        v=hsv[:,:,2]  # get brightness channel

        if d==index:
            drange[1]=v.max()  # get original brightness max of reference
            drange[0]=v.min()  # get original brightness min of reference
        
        # skimage.io.imsave('b'+str(d+1)+'.tif',v)
        tmp=np.multiply(fftpack.fftshift((fftpack.fft2(v))),low_pass)
        tmp=np.real(fftpack.ifft2(fftpack.ifftshift(tmp)))
        tmp=np.maximum(0,np.minimum(tmp,255))
        # skimage.io.imsave('a'+str(d+1)+'.tif',tmp.astype(np.uint8))
    
        v=np.float32(np.absolute(np.float32(v)-np.float32(tmp)))  # image division or image subtraction
        # v=np.float32(v/tmp)
        # v=v+np.float32(np.mean(tmp))
        # v=cv2.normalize(v,None,0,255,cv2.NORM_MINMAX,cv2.CV_32F)  # normalize intensity individually
        
        tmp=np.float32(hsv)
        tmp[:,:,2]=v
        inter_result.append(tmp)
        
    vmax=inter_result[index][:,:,2].max()  # get current brightness max of reference
    vmin=inter_result[index][:,:,2].min()  # get current brightness min of reference
    for d in tqdm.tqdm(range(numI),desc='    starting intensity normalization...',position=0,leave=True):
        inter_result[d][:,:,2]=(inter_result[d][:,:,2]-vmin)*((drange[1]-drange[0])/(vmax-vmin))+drange[0]  # normalize all frames based on current reference to original reference
        tmp=np.maximum(0.0,np.minimum(inter_result[d],255.0))  # confine data to [0,255]
        tmp=np.uint8(np.floor(tmp))  # convert to uint8
        result.append(cv2.cvtColor(tmp,cv2.COLOR_HSV2RGB))  # convert from hsv to rgb
    
    return result




# %%  main function
if __name__=="__main__": 
    # add input arguments and parse it to an object
    parser=argparse.ArgumentParser(description='MUSE-SCAN processing')
    parser.add_argument('--filepath',metavar='C:/Users/...',help='file root path')
    parser.add_argument('--block',type=int,default=51,metavar='x(odd)',help='focus window size')
    parser.add_argument('--overlap',type=int,default=5,metavar='n(%)',help='overlap percentage')
    parser.add_argument('--mosaic',type=int,nargs=2,default=[3,3],metavar=('row','col'),help='mosaic tile count')
    parser.add_argument('--shading',type=int,default=0,metavar='n',help='shading correction reference')
    parser.add_argument('--contrast',action='store_true',help='flag for contrast enhancement')
    
    # parse and read arguments
    args=parser.parse_args()
    rootpath=args.filepath
    block=args.block    # by->row, bx-col
    overlap=args.overlap
    row,col=args.mosaic
    if args.shading is not None:  # check if shading correction is requested
        shading_flag=True
        shading_reference=args.shading
    else:
        shading_correction=False
    contrast_flag=args.contrast  # check if contrast enhancement is requested
        
    
    # sort all folders in root path
    (_,folders,files)=next(os.walk(rootpath))    # get the tile stacks, ground truth images, and log file
    folders=[i for i in folders if 'result' not in i.lower()] # ignore previous result folder
    fformat=files[0].split('.')[1]  # get the file suffix -- format
    numF=len(folders)   # get tile count
    log=files[-1]  # get the log file
    
    # create output folder if not exist
    outpath=rootpath+'/result'  # folder for all results
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    Tpath=outpath+'/T'  # folder for refocused test case
    if not os.path.exists(Tpath):
        os.mkdir(Tpath)
    Hpath=outpath+'/H'  # folder for refocused height map
    if not os.path.exists(Hpath):
        os.mkdir(Hpath)
        
    # start batch refocusing
    t_start=time.time() # start timer
    stdout('\n**********starting batch refocusing**********')
    for d in range(numF):
        stdout('\nstarting refocusing stack #'+str(d+1)+' out of '+str(numF)+':')
        # tmp1,tmp2=load_stack(rootpath+'/'+folders[d],fformat)
        # tmp1,tmp2=load_stack_parallel(rootpath+'/'+folders[d],fformat)
        tmp1,tmp2=load_stack_gpu(rootpath+'/'+folders[d],fformat)
        # result,_,hmap=refocus(tmp1,tmp2,block)
        result,_,hmap=refocus_gpu(tmp1,tmp2,block)
        skimage.io.imsave(Tpath+'/{:03d}.'.format(d+1)+fformat,result,check_contrast=False)
        skimage.io.imsave(Hpath+'/{:03d}.'.format(d+1)+fformat,hmap,check_contrast=False)
        
    # start stitching for test case T
    stdout('\n\nstarting stitching test case without shading correction:')
    # T,_=load_stack(Tpath,fformat)
    # T,_=load_stack_parallel(Tpath,fformat)
    T,_=load_stack_gpu(Tpath,fformat)
    # sT=stitch_snake(T,row,col,overlap)  # T as PIL.Image
    sT=stitch(T,row,col,overlap)  # T as PIL.Image
    sT.save(outpath+'/T.'+fformat)
    # skimage.io.imsave(outpath+'/T.'+fformat,sT)
    if contrast_flag:
        enhance_T=contrast_enhance(T)  # contrast enhancement
        # enhance_sT=stitch_snake(enhance_T,row,col,overlap)  # enhance_T as PIL.Image
        enhance_sT=stitch(enhance_T,row,col,overlap)  # enhance_T as PIL.Image
        enhance_sT.save(outpath+'/enhance_T.'+fformat)
    
    # start stitching for test case T with shading correction if requested
    if shading_flag:
        stdout('\n\nstarting stitching test case with shading correction:')
        pTpath=outpath+'/pT'  # folder for corrected test case，"p" stands for processing
        if not os.path.exists(pTpath):
            os.mkdir(pTpath)
        # pT=shade_correct_intensity_lowpass(T,10,reference)
        pT=shade_correct_intensity(T,300,shading_reference)  # shading correction
        for d in range(len(pT)):
            skimage.io.imsave(pTpath+'/{:03d}.'.format(d+1)+fformat,pT[d].astype(np.uint8),check_contrast=False)
        # spT=stitch_snake(pT,row,col,overlap)  # pT as PIL.Image
        spT=stitch(pT,row,col,overlap)  # pT as PIL.Image
        spT.save(outpath+'/pT.'+fformat)
        if contrast_flag:
            enhance_pT=contrast_enhance(pT)
            # enhance_spT=stitch_snake(enhance_pT,row,col,overlap)  # enhance_pT as PIL.Image
            enhance_spT=stitch(enhance_pT,row,col,overlap)  # enhance_pT as PIL.Image
            enhance_spT.save(outpath+'/enhance_pT.'+fformat)
    
    # start stitching for height map H
    stdout('\n\nstarting stitching height map:')
    # H,_=load_stack(Hpath,fformat)
    # H,_=load_stack_parallel(Hpath,fformat)
    H,_=load_stack_gpu(Hpath,fformat)
    # sH=stitch_snake(H,row,col,overlap,'average')  # H as PIL.Image
    sH=stitch(H,row,col,overlap)  # H as PIL.Image
    sH.save(outpath+'/H.'+fformat)
        
    # start stitching for ground truth GT
    stdout('\n\nstarting stitching ground truth:')
    # GT,_=load_stack(rootpath,fformat)
    # GT,_=load_stack_parallel(rootpath,fformat)
    GT,_=load_stack_gpu(rootpath,fformat)
    # pGT=shade_correct_intensity(GT,300,shading_reference)  # shading correction
    # spGT=stitch(pGT,row,col,overlap)  # pT as PIL.Image
    # spGT.save(outpath+'/pGT.'+fformat)
    # if contrast_flag:
    #     enhance_pGT=contrast_enhance(pGT)
    #     # enhance_spT=stitch_snake(enhance_pT,row,col,overlap)  # enhance_pT as PIL.Image
    #     enhance_spGT=stitch(enhance_pGT,row,col,overlap)  # enhance_pT as PIL.Image
    #     enhance_spGT.save(outpath+'/enhance_pGT.'+fformat)
    # sGT=stitch_snake(GT,row,col,overlap)  # GT as PIL.Image
    sGT=stitch(GT,row,col,overlap)  # GT as PIL.Image
    sGT.save(outpath+'/GT.'+fformat)
    
    # update log file with relative actual z-drive range acquired from generated height map
    tmp=np.asarray(sH)  # convert to numpy.array for sorting
    with open(rootpath+'/'+log,'a') as logf:
        logf.write('\nrelative actual z-drive range: [%d, %d] in slice' % (tmp.min(),tmp.max()))
    
    t_end=time.time() # end timer
    print('\n\ntime to process: '+str(round(t_end-t_start,3))+'s\n')
# %%