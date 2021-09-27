# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:13:02 2021

@author: Jack
"""

import skimage.io
import skimage.color
import cv2
import tqdm
import scipy.ndimage
import os
import sys
import glob
import time
import shutil
import numpy as np
from PIL import Image
import PIL.ImageOps


# align two images (im2 -> im1) using enhanced correlation coefficient maximization
# fullscale ECC algorithm assuming translation only
# input: image #1, image #2, max filter power, kernal sigma
# output: warp
def align_two_full(im1,im2,max_power=5,sigma=3):
    warp_mode=cv2.MOTION_TRANSLATION # set motion to translation
    number_of_iterations=1000 # specify number of iterations
    termination_eps=1e-6 # specify the threhold of the increment in the correlation coefficient between two iterations
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,number_of_iterations,termination_eps) # define termination criteria
    
    if im1.ndim==3: # convert to grayscale
        im1=skimage.color.rgb2gray(im1.astype(np.float32))
        im2=skimage.color.rgb2gray(im2.astype(np.float32))
    
    i = 3 # initial filter power, larger translation needs larger i
    while i<=max_power: # attempt to fit the image, increasing the blur as necessary  
        im1_blur=scipy.ndimage.filters.gaussian_filter(im1,(2**i,2**i)) # no need for this since cv2. findTransformECC has option for gaussian filtering
        im2_blur=scipy.ndimage.filters.gaussian_filter(im2,(2**i,2**i))
        # im1_blur=scipy.ndimage.uniform_filter1d(im1,2**(i+1),axis=0) # in case gaussian blurring is slow, use uniform filtering
        # im1_blur=scipy.ndimage.uniform_filter1d(im1,2**(i+1),axis=1)
        # im2_blur=scipy.ndimage.uniform_filter1d(im2,2**(i+1),axis=0)
        # im2_blur=scipy.ndimage.uniform_filter1d(im2,2**(i+1),axis=1)
        
        warp=np.eye(2,3,dtype=np.float32) # initial the 2X3 warp matrix
        try:
            (cc,warp)=cv2.findTransformECC(im1_blur,im2_blur,warp,warp_mode,criteria,None,sigma) # kernel=sigma*2+1
        except:
            i=i+1
        else:
            break

    return warp


# pyramid ECC algorithm assuming translation only
# input: image #1, image #2, kernel sigma
# output: warp
def align_two_pyramid(im1,im2,sigma=3):
    warp_mode=cv2.MOTION_TRANSLATION # set motion to translation
    number_of_iterations=1000 # specify number of iterations
    termination_eps=1e-6 # specify the threhold of the increment in the correlation coefficient between two iterations
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,number_of_iterations,termination_eps) # define termination criteria
    
    if im1.ndim==3: # convert to grayscale
        im1=skimage.color.rgb2gray(im1.astype(np.float32))
        im2=skimage.color.rgb2gray(im2.astype(np.float32))
        
    lvl=int(np.ceil((im1.shape[0]/500)/2)) # compute pyramid level (image pixel less than 500 should be good)
    
    pyr_im1=[im1] # initial pyramid for im1
    pyr_im2=[im2] # initial pyramid for im2
    for l in range(lvl): # create pyramid
        pyr_im1.insert(0,cv2.resize(pyr_im1[0],None,fx=1/2,fy=1/2,interpolation=cv2.INTER_AREA)) # downscale current level by 2
        pyr_im2.insert(0,cv2.resize(pyr_im2[0],None,fx=1/2,fy=1/2,interpolation=cv2.INTER_AREA))
    
    warp=np.eye(2,3,dtype=np.float32) # initial the 2X3 warp matrix
    for l in range(lvl): # run pyramid ECC
        cc,warp=cv2.findTransformECC(pyr_im1[l],pyr_im2[l],warp,warp_mode,criteria,None,sigma) # intrinsic gaussian filtering
        warp=warp*np.array([[1,1,2],[1,1,2]],dtype=np.float32) # upscale prediction by 2
    
    return warp


# auto find crop matrix for aligned image
# input: aligned image, preset color
# output: crop matrix
def fit(im,color=(0,0,0)):
    color_area=(im[:,:,0]==color[0])&(im[:,:,1]==color[1])&(im[:,:,2]==color[2]) # locate preset color
    im[color_area]=(0,0,0) # replace preset color to black
    # gim=skimage.color.rgb2gray(im.astype(np.float32)) 
    tmp=Image.fromarray(im).convert('L') # convert to PIL.Image and grayscale
    crop=tmp.getbbox() # get valid (non-black) region
    
    return crop


# read the file name given full absolute name
def read_name(path):
    path=path.replace('\\','/') # get rid of ambiguity
    slist=path.split('/') # split the full name
    result=slist[-1] # read the image name
    
    return result


# align a stack of images (for muse-milling)
# the accumulated misalignment should not exceed the size of the original image
def align_stack(path='test',fmt='tif',sigma=3,method='pyramid',mode='none',color=(0,0,0)):    
    filemask=path+'/*.'+fmt # create filemask
    files=glob.glob(filemask) # sort images
    numI=len(files)
    
    out_dir=path+'/aligned' # create the path for alignment
    if not os.path.isdir(out_dir): # create a folder if not exist
        os.mkdir(out_dir)
    
    warps=[np.eye(2,3,dtype=np.float32)] # initial translation matrices
    shutil.copy2(files[0],out_dir) # copy the first frame (reference)
    # im1=skimage.io.imread(files[0]) # read the first frame (reference)
    # skimage.io.imsave(out_dir+"/001.tif",im1,check_contrast=False) # save the first frame
    for i in tqdm.tqdm(range(1,numI),desc='aligning images...'):
        im1=skimage.io.imread(files[i-1]) # read the last frame
        im2=skimage.io.imread(files[i]) # read current frame
        
        if method=='pyramid': # use pyramid ECC
            warps.append(align_two_pyramid(im1,im2,sigma)) # align im2 to im1
        else: # use fullscale ECC
            warps.append(align_two_full(im2,im2,sigma)) # align im2 to im1
            
        warps[i][0,2]=warps[i][0,2]+warps[i-1][0,2] # accumulate translation
        warps[i][1,2]=warps[i][1,2]+warps[i-1][1,2]
        
        out=cv2.warpAffine(im2,warps[i],(im2.shape[1],im2.shape[0]),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,borderValue=(color[0],color[1],color[2]))  # apply translation
        suf=read_name(files[i]) # read image name
        skimage.io.imsave(out_dir+'/'+suf,out,check_contrast=False)
    
    im1=skimage.io.imread(files[0])
    ix=im1.shape[1] # get original dimensions
    iy=im1.shape[0]
    if mode!='none':
        _filemask=out_dir+'/*'+fmt # update filemask
        _files=glob.glob(_filemask) # update images
        process_dir=path+'/'+mode # create the path for crop
        if not os.path.isdir(process_dir): # create a folder if not exist
            os.mkdir(process_dir)
            
        if mode=='crop': # crop image to fit
            regions=[[],[],[],[]] # initial a valid region matrix for each image
            for i in tqdm.tqdm(range(numI),desc='fitting images...'): # find valid regions, Rect -> [x0,y0,x1,y1]
                im=skimage.io.imread(_files[i]) # read current frame 
                tmp=fit(im,color) # fit current frame
                for k in range(4):
                    regions[k].append(tmp[k]+(1 if k<2 else -1)) # copy region values, compensate for interpolation
            region=[] # initial the final region matrix considering all images
            region.append(max(regions[0])) # left x0
            region.append(max(regions[1])) # top y0
            region.append(min(regions[2])) # right x1
            region.append(min(regions[3])) # bottom y1
            for i in tqdm.tqdm(range(numI),desc='cropping images...'):
                im=skimage.io.imread(_files[i]) # read
                im=im[region[1]:region[3],region[0]:region[2],] # crop
                suf=read_name(_files[i]) # read image name
                # im=scipy.ndimage.median_filter(im,size=2,mode='mirror') # use size=2 median filter to remove hot pixels - NOT FAST!
                skimage.io.imsave(process_dir+'/'+suf,im,check_contrast=False) # save
                
        elif mode=='extend': # extend image to fit
            expand=[] # [left x0,top y0,right x1,bottom y1]
            expand.append(np.array(warps)[:,0,2].max()) # get left expansion
            expand.append(np.array(warps)[:,1,2].max()) # get top expansion
            expand.append(np.array(warps)[:,0,2].min()) # get right expansion
            expand.append(np.array(warps)[:,1,2].min()) # get bottom expansion
            ox=ix+int(np.floor(expand[0])-np.ceil(expand[2])) # get fitted dimensions
            oy=iy+int(np.floor(expand[1])-np.ceil(expand[3]))
            shift=np.zeros((2,3),dtype=np.float32) # initial reference shift
            shift[0,2]=expand[0]
            shift[1,2]=expand[1]
            for i in tqdm.tqdm(range(numI),desc='extending images...'):
                im=skimage.io.imread(files[i]) # read
                out=np.ones((oy,ox,3),np.uint8)
                out=cv2.warpAffine(im,warps[i]-shift,(ox,oy),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,borderValue=(color[0],color[1],color[2])) # extend
                suf=read_name(files[i]) # read image name
                skimage.io.imsave(process_dir+'/'+suf,out,check_contrast=False) # save
            
            # ox=ix+region[0]+(ix-region[2]) # get fitted dimensions
            # oy=iy+region[1]+(iy-region[3])    
            # this method misses sub-pixel offset
            # for i in tqdm.tqdm(range(numI),desc='extending images...'):
            #     im=skimage.io.imread(files[i]) # read
            #     out=np.zeros((oy,ox,3),dtype=np.uint8) # initial
            #     shift=[iy-region[3],ix-region[2]] # shift value, initial to KINDA center
            #       # update along x-axis
            #     if regions[0][i]==0: # move left
            #         shift[1]=shift[1]-(ix-regions[2][i])
            #     else: # move right
            #         shift[1]=shift[1]+regions[0][i]
            #     # update along y-axis
            #     if regions[1][i]==0: # move up
            #         shift[0]=shift[0]-(iy-regions[3][i])
            #     else: # move down
            #         shift[0]=shift[0]+regions[1][i]
            #     out[shift[0]:shift[0]+iy,shift[1]:shift[1]+ix,:]=im # extend
            #     skimage.io.imsave(process_dir+'/{:03d}.tif'.format(i+1),out,check_contrast=False) # save
            # for i in tqdm.tqdm(range(numI),desc='extending images...'):
            #     im=skimage.io.imread(files[i]) # read
            #     center=np.zeros((2,3))
            #     center[0,2]=center[0,2]+ix-region[2]
            #     center[1,2]=center[1,2]+iy-region[3]
            #     out=cv2.warpAffine(im,warps[i]-center,(ox,oy),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP) # extend
            #     suf=read_name(files[i]) # read image name
            #     # out=scipy.ndimage.median_filter(out,size=2,mode='mirror') # use size=2 median filter to remove hot pixels - NOT FAST!
            #     skimage.io.imsave(process_dir+'/'+suf,out,check_contrast=False) # save
                
    return warps[-1] # return last warp
   

# align multiple stacks individually (for muse-milling)
def align_multiple_stacks(path='test',fmt='tif',method='pyramid',process='none',skip='false'):
    (_,folders,_)=next(os.walk(path)) # sort stacks
    numF=len(folders)
    paths=[path+'/'+f for f in folders] # get individual directories
    
    for f in range(numF): # align each stack individually
        sys.stdout.write('\n'+'processing stack #'+str(f+1)+':\n')
        sys.stdout.flush()
        align_stack(paths[f],fmt,method,process,skip) # align each stack
        

# align series stacks (exclusively for Dilani)
# first align each stack individually, then align rest to reference, finally superimpose to one stack
# assume negligible drift happen during stacks
def align_series_stacks(path='test',fmt='tif',method='pyramid'):
    (_,folders,_)=next(os.walk(path)) # sort stacks
    folders=[i for i in folders if 'result' not in i.lower()] # ignore previous result folder
    numF=len(folders)
    paths=[path+'/'+f for f in folders] # get individual directories
    
    warps=[np.eye(2,3,dtype=np.float32)] # initial translation matrices
    for f in range(numF): # first align each stack individually
        sys.stdout.write('\n'+'processing stack #'+str(f+1)+':\n')
        sys.stdout.flush()
        warps.append(align_stack(paths[f],fmt,method)) # align each stack
        
        # current warp is used to align next frame, so warps[0] -> [[1,0,0],[0,1,0]]
        warps[f+1][0,2]=warps[f+1][0,2]+warps[f][0,2] # accumulate translation
        warps[f+1][1,2]=warps[f+1][1,2]+warps[f][1,2]
    
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    shutil.copytree(paths[0]+'/aligned',paths[0]+'/realigned',dirs_exist_ok=True) # copy the first stack (reference)
    for f in tqdm.tqdm(range(1,numF),desc='realigning stacks...'): # then align rest to reference
        filemask=paths[f]+'/aligned/*.'+fmt # create filemask  
        files=glob.glob(filemask) # sort images
        numI=len(files)
    
        align_dir=paths[f]+'/realigned' # create the path for realignment
        if not os.path.isdir(align_dir): # create a folder if not exist
            os.mkdir(align_dir)
        
        for i in range(numI): # realign each frame to reference
            im=skimage.io.imread(files[i]) # read
            out=cv2.warpAffine(im,warps[f],(im.shape[1],im.shape[0]),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)  # translation
            skimage.io.imsave(align_dir+'/{:03d}.tif'.format(i+1),out,check_contrast=False) # save
    
    # finally superimpose to one stack
    add_dir=path+'/result' # create the path for superimposition
    if not os.path.isdir(add_dir): # create a folder if not exist
        os.mkdir(add_dir)
    
    regions=[[],[],[],[]] # initial region matrix [x0,y0,x1,y1], each stack generates one
    region=[] # initial final region matrix [x0,y0,x1,y1]
    for f in tqdm.tqdm(range(numF),desc='fitting stacks...'): # fit each stack
        filemask=paths[f]+'/realigned/*.'+fmt # update filemask  
        files=glob.glob(filemask) # sort images
        numI=len(files)
        
        crops=[[],[],[],[]] # initial crop matrix [x0,y0,x1,y1]
        for i in range(numI):
            im=skimage.io.imread(files[i]) # read current frame  
            tmp=fit(im) # fit current frame
            for k in range(4):
                crops[k].append(tmp[k]) # copy fit values
        # get region for each stack
        regions[0].append(max(crops[0])) # left x0
        regions[1].append(max(crops[1])) # top y0
        regions[2].append(min(crops[2])) # right x1
        regions[3].append(min(crops[3])) # bottom y1
    
    # get final region for all stacks
    region.append(max(regions[0])) # left x0
    region.append(max(regions[1])) # top y0
    region.append(min(regions[2])) # right x1
    region.append(min(regions[3])) # bottom y1
    
    # make sure that the images are not more than a thousand, otherwise change to 4-digit
    for i in tqdm.tqdm(range(numI),desc='superimposing images...'): # superimpose stacks
        tmp=np.zeros((region[3]-region[1],region[2]-region[0],3),np.uint8) # initial an empty image
        for f in range(numF):
            im=skimage.io.imread(paths[f]+'/realigned/{:03d}'.format(i+1)+'.'+fmt) # read
            im=im[region[1]:region[3],region[0]:region[2],:] # crop
            tmp=cv2.add(tmp,im) # superimposition
        skimage.io.imsave(add_dir+'/{:03d}.tif'.format(i+1),tmp,check_contrast=False) # save


# pull separate channel for visualization, default to red channel (0)
def separate_channel(path='test',fmt='tif',channel=0,invert=False):
    filemask=path+'/*.'+fmt # create filemask
    files=glob.glob(filemask) # sort images
    numI=len(files)
    
    out_dir=path+'/ch'+str(channel) # create the path for separate channel
    if not os.path.isdir(out_dir): # create a folder if not exist
        os.mkdir(out_dir)
        
    for i in tqdm.tqdm(range(numI),desc='separating images...'):
        im=skimage.io.imread(files[i])[:,:,channel] # load image and separate channel
        tmp=Image.fromarray(im) # convert to PIL.Image
        if invert==True:
            tmp=PIL.ImageOps.invert(tmp) # invert color
        tmp=np.array(tmp) # convert to numpy.array
        skimage.io.imsave(out_dir+'/{:03d}.tif'.format(i+1),tmp,check_contrast=False)


# volume projection using maximum or minimum
def projection(path='test',fmt='tif',mode='maximum',frame=(0,10)):
    filemask=path+'/*.'+fmt # create filemask
    files=glob.glob(filemask) # sort images
    numI=len(files)
    
    if frame[0]<0 or frame[1]>numI: # avoid segmentation fault
        return
    
    color=True # is rgb
    tmp=skimage.io.imread(files[0]) # load first frame
    if len(tmp.shape)==2: # determine color space
        color=False
    
    I=[] # rgb images
    gI=[] # grayscale images
    for i in tqdm.tqdm(range(frame[0],frame[1]),desc='loading images...'):
        I.append(skimage.io.imread(files[i])) # load image
        if color:
            gI.append(skimage.color.rgb2gray(I[i].astype(np.float32))) # copy grayscale
        else:
            gI.append(I[i])
    
    sx=I[0].shape[1]
    sy=I[0].shape[0]
    if color:
        out=np.zeros((sy,sx,3),np.uint8) # rgb output image
    else:
        out=np.zeros((sy,sx),np.uint8) # grayscale output image
    for i in tqdm.tqdm(range(frame[0],frame[1]),desc='projecting images...'):
        tmp=np.asarray(gI) # cast to array
        if mode=='maximum':
            m=np.uint8(tmp.argmax(axis=0)) # find maximum
        elif mode=='minimum':
            m=np.uint8(tmp.argmin(axis=0)) # find minimum
        else:
            m=np.uint8(tmp.argmax(axis=0)) # default to maximum
        
    for iy in range(sy):
        for ix in range(sx):
            if color:
                out[iy,ix,:]=I[m[iy,ix]][iy,ix,:] # copy positions
            else:
                out[iy,ix]=I[m[iy,ix]][iy,ix] # copy positions
            
    skimage.io.imsave(mode+'.tif',out,check_contrast=False) # save to the directory of the script
    


# %%  main function
t_start=time.time() # start timer
align_stack('test/001',mode='extend',color=(255,255,255))
# separate_channel('test/001/extend',channel=0,invert=True)
# projection('test/001/aligned/ch0',mode="maximum",frame=(0,201))
# align_multiple_stacks('test')
# align_series_stacks('test')
t_end=time.time() # end timer
print('\ntime to process: '+str(round(t_end-t_start,3))+'s\n')