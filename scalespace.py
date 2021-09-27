# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 12:39:12 2020

@author: JACK
"""


import skimage.io
import skimage.filters
import skimage.feature
import scipy.ndimage
import scipy.io
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
# from mayavi import mlab
import os
import sys
import glob
import tqdm
import time


# load stack function
# input: file mask fm
# output: grayscale stack
def load_stack(fm):
    fs=glob.glob(fm)    # sort all images
    numI=len(fs)    # count all images
    I=[]    # initialize image stack
    
    for i in tqdm.tqdm(range(numI),desc='  loading images...',position=0,leave=True):
        tmpI=skimage.io.imread(fs[i])   # read an image as is
        # if tmpI.dtype==np.uint16:
        #     tmpI=(tmpI/256).astype(np.uint8)
        I.append(tmpI)    # append all images as reference
        if I[i].ndim==3:    # convert rgb to grayscale
            tmpI=0.2989*I[i][:,:,0]+0.5870*I[i][:,:,1]+0.1140*I[i][:,:,2]    # using rec601 luma coding
            I[i]=np.round(tmpI).astype(I[i].dtype)   # keep its original bit depth
            
    return I


# variance map function
# input: image s, window size w (window size at least 3x3 and odd)
# output: variance v at each pixel
# "variance trick": sum of square - square of sum
### to-do: create a way to do the non-padding convolution instead of manual cropping
def var_map(s,w):
    o=np.int8(np.floor(w/2)) # offset to left-top origin
    
    mean=scipy.ndimage.uniform_filter(np.float32(s),w,origin=(-o,-o))   # get mean
    sqr_mean=scipy.ndimage.uniform_filter(np.float32(s)**2,w,origin=(-o,-o))    # get square mean
    var=sqr_mean-mean**2
    var=var[0:var.shape[0]-w+1,0:var.shape[1]-w+1]  # manual cropping -- data to the left top corner
        
    return var


# normalized variance map function
# input: image s, window size w (window size at least 3x3 and odd)
# output: normalized variance v at each pixel
# the best focus measure for microscopy [sun2004autofocusing]
def norm_var_map(s,w):
    o=np.int8(np.floor(w/2)) # offset to left-top origin
    
    mean=scipy.ndimage.uniform_filter(np.float32(s),w,origin=(-o,-o))   # get mean
    sqr_mean=scipy.ndimage.uniform_filter(np.float32(s)**2,w,origin=(-o,-o))    # get square mean
    var=sqr_mean-mean**2
    
    # mu=s.mean()  # get overall intensity mean
    var=var/mean  # normalization to compensate for the differences in average image intensity among different images
    
    var=var[0:var.shape[0]-w+1,0:var.shape[1]-w+1]  # manual cropping
        
    return var


# weighted variance map function
# input: image s, window size w, weight template t
# output: weighted variance at each pixel
def weighted_var_map(s,w,t):
    o=np.int8(np.floor(w/2)) # offset to left-top origin
    
    sx=I[0].shape[1]    # get col
    sy=I[0].shape[0]    # get row
    var=np.zeros((sy-2*o,sx-2*o),np.float32)
    
    for i in range(o,sy-o):
        for j in range(o,sx-o):
            tmps=s[i-o:i+o+1,j-o:j+o+1]     # get the processing matrix
            mean=np.average(tmps,weights=t)     # get weighted mean
            var[i-o,j-o]=np.average((tmps-mean)**2,weights=t)    # get weighted variance
            
    return var
            

# height map function
# input: stack ss
# output: height map
def hei_map(ss):
    tmpss=np.asarray(ss)
    h=np.uint16(tmpss.argmax(axis=0))  # find largest variance position
    
    return h


# selective variance output function
# input: variance list v, window index w, image index i
# output: selective variance image
def get_var(v,w,i):
    tmp=np.round(v[w][i])
    tmp=cv.normalize(tmp,None,0,255,cv.NORM_MINMAX,cv.CV_8UC1)  # cast to uint8
    skimage.io.imsave(filepath+'/result/variance_i'+str(i+1)+'_w'+str((w+1)*2+1)+'.tif',np.uint8(tmp),check_contrast=False)


# detect spikes as local extrema with a fixed global threshold
# input: height map h, fixed global threshold t
# output: threshold height map th
def find_spike_fixed(h,t):
    th=[]
    for i in range(len(h)):
        tmp=skimage.feature.peak_local_max(h[i],(i+1),threshold_abs=t)
        th.append(tmp)
        
    return th


# Gaussian second derivative function
# input: height map h, sigma s
# output: Laplace matrix
def get_gaussian_laplace(h,s=3):
    glp=scipy.ndimage.gaussian_laplace(h,sigma=s)
    
    return glp


# approximate second derivative function
# input: height map h, finite difference accuracy a
# output: Laplace matrix
# accuracy:
# 2					1	−2	1				
# 4				−1/12	4/3	−5/2	4/3	−1/12			
# 6			1/90	−3/20	3/2	−49/18	3/2	−3/20	1/90		
# 8		−1/560	8/315	−1/5	8/5	−205/72	8/5	−1/5	8/315	−1/560
def get_laplace(h,a=2):
    if a==2:
        c=np.array([1,-2,1],np.float16)
    elif a==4:
        c=np.array([-1/12,4/3,-5/2,4/3,-1/12],np.float16)
    elif a==6:
        c=np.array([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90],np.float16)
    else:
        c=np.array([-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560],np.float16)
    
    def derivative2(input,axis,output,mode,cval):
        return scipy.ndimage.correlate1d(input,c,axis,output,mode,cval,0)
    
    lp=scipy.ndimage.generic_laplace(h,derivative2,None,'reflect',0.0)
    
    return lp


# erode-dilate function, dilation of erosion (morphological opening -- remove small objects)
# input: original height map h, kernel size k, template t
# output: modified height map
# template can be binary square, cross, circle
def erode_dilate(h,k=3,t='square'):
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
    tmph=cv.erode(h,kernel,iterations=1)    # erode away boundary spikes
    edh=cv.dilate(tmph,kernel,iterations=1)   # connect diminished components
        
    return edh


# erode-dilate function, erosion of dilation (morphological closing -- remove small holes)
# input: original height map h, kernel size k, template t
# output: modified height map
# template can be binary square, cross, circle
def dilate_erode(h,k=3,t='square'):
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
    tmph=cv.dilate(h,kernel,iterations=1)    # erode away boundary spikes
    edh=cv.erode(tmph,kernel,iterations=1)   # connect diminished components
        
    return edh


# output stack function
# input: original stack ss, height map h, offset o
# output: output stack
def get_stack(ss,h,o):
    ssx=h.shape[1]
    ssy=h.shape[0]
    sst=ss[0].dtype
    out=np.zeros((ssy,ssx),sst)
    
    for i in range(ssy):
        for j in range(ssx):
            out[i,j]=ss[h[i,j]][i+o,j+o]
    
    return out


# save surface function
# input: height map h, image count c, output name n, elev e, azim a
# output: surface image
def save_surface(h,c,n,e=45,a=30):
    hx=h.shape[1]
    hy=h.shape[0]
    
    fig=plt.figure()
    ax=Axes3D(fig)
    X=np.arange(0,hx,1)
    Y=np.arange(0,hy,1)
    X,Y=np.meshgrid(X,Y)
    surf=ax.plot_surface(Y,X,h+1,cmap=cm.coolwarm)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.set_xlim(0,hy)
    ax.set_ylim(0,hx)
    ax.set_zlim(1,c)
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.view_init(elev=e,azim=a)
    fig.savefig(n)
    plt.close()
    
    
# plot surface function
# input: height map h, figure identifier f, image count c, elev e, azim a
# output: surface image
# output: surface plot
def plot_surface(h,f,c,e=45,a=30):
    hx=h.shape[1]
    hy=h.shape[0]
    
    fig=plt.figure(f)
    ax=Axes3D(fig)
    X=np.arange(0,hx,1)
    Y=np.arange(0,hy,1)
    X,Y=np.meshgrid(X,Y)
    surf=ax.plot_surface(Y,X,h+1,cmap=cm.coolwarm)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.set_xlim(0,hy)
    ax.set_ylim(0,hx)
    ax.set_zlim(1,c)
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    ax.view_init(elev=e,azim=a)
    plt.show()


def stdout(words):
    sys.stdout.write(str(words)+'\n')
    sys.stdout.flush()




path='tmp/*.tif'
I=load_stack(path)
# M=scipy.io.loadmat('tmp/data.mat')
# shift=M['shift']
for i in range(len(I)):
    tmp=np.uint8(I[i]-10)
    # tmp=np.uint16(I[i]-shift[0][2])
    skimage.io.imsave('tmp/TH_'+str(i+1).zfill(3)+'.tif',tmp,check_contrast=False)









# main script here
rootpath='C:/Users/Jack/Desktop/test'   # root folder to process
folders=os.listdir(rootpath)    # detect all folders
# matFile=[i for i in folders if 'mat' in i.lower()]  # get the mat file
# matFile=os.path.join(rootpath,matFile[0]).replace('\\','/')  
folders=[i for i in folders if 'tif' not in i.lower()]    # delete previous saved tif files
# folders=[i for i in folders if 'mat' not in i.lower()]    # delete mat files

GT=[]
GTH=[]
T=[]
TH=[]

t_start=time.time()     # start timer
for f in range(len(folders)):   # start batch process
    stdout('\nprocessing stack #'+str(f+1))
    filepath=os.path.join(rootpath,folders[f]).replace('\\','/')  
    # filepath=rootpath+'/'+folders[f]
    # filepath='C:/Users/Jack/Desktop/deposition'  # file path
    fformat='.tif'     # file format
    
    I=load_stack(filepath+'/*'+fformat)  # load the image stack
    Ix=I[0].shape[1]   # get image width, [1]->col->X
    Iy=I[0].shape[0]   # get image height, [0]->row->Y
    numI=len(I)    # get image count
    typeI=I[0].dtype    # get image type
    
    level=5   # pyramid layer count (at least 1), each layer adopts window size level*2+1
    out_var=False    # flag for saving variance image
    var_id=0    # selective variance index -> layer
    plt_hm=False    # flag for plotting height map
    hm_id=4   # selective plot index -> layer
    flp_hm=True    # flag for flipping height map at even order, this is only for Dr. Shan's test data
    ed_hm=True    # flag for processing erode-dilation
    rd_hm=False    # flag for generating the Laplace matrix of height map
    
    outpath=os.path.join(filepath,'result').replace('\\','/')
    # outpath=filepath+'/result'
    if not os.path.exists(outpath):  # create a output folder if not exist
        os.mkdir(outpath)

    #     pyramid
    #    *layer 4*    -> [image 1,image 2,image 3,...]
    #   **layer 3**   -> [image 1,image 2,image 3,...]
    #  ***layer 2***  -> [image 1,image 2,image 3,...]
    # ****layer 1**** -> [image 1,image 2,image 3,...]  
    V=[[] for _ in range(level)]   # initialize a pyramid for variance matrix
    for d in tqdm.tqdm(range(numI),desc='  calculating variance...',position=0,leave=True):    # build a pyramid for each image
        for l in range(level):      # build each layer       
            V[l].append(norm_var_map(I[d],(l+1)*2+1))  # output as float32
            #locals()['var'+str(p)]=var_map(I,3**p)
    if out_var==True:    # output the bottome layer of the variance pyramid
        varpath=os.path.join(outpath,'variance'+str(var_id+1).zfill(3)).replace('\\','/')
        if not os.path.exists(varpath):
            os.mkdir(varpath)
        for d in tqdm.tqdm(range(numI),desc='  saving example variance...',position=0,leave=True):
            tmpV=cv.normalize(V[var_id][d],None,0,65535,cv.NORM_MINMAX,cv.CV_16U)  # normalize variance to [0,65535] as uint16
            skimage.io.imsave(varpath+'/'+str(d+1).zfill(3)+fformat,tmpV,check_contrast=False)
    
    H=[]   # initialize a pyramid for height map
    edH=[]
    LPH=[]
    LPedH=[]
    for l in tqdm.tqdm(range(level),desc='  generating height map...',position=0,leave=True):    # create a height map for each layer 
        H.append(hei_map(V[l]))     # output as uint16
        
        if flp_hm==True:
            if (f%2)==1:    # flip even folder temporarily for processing
                H[l]=numI-1-H[l]
                
        # H[l]=cv.normalize(H[l],None,0,numI-1,cv.NORM_MINMAX,cv.CV_16U)
        skimage.io.imsave(outpath+'/hmap'+str(l+1).zfill(3)+fformat,H[l]+1,check_contrast=False)   # add 1 to reflect actual index
        np.savetxt(outpath+'/hmap'+str(l+1).zfill(3)+'.txt',H[l]+1,fmt='%i')          
        if rd_hm==True:
            LPH.append(get_gaussian_laplace(H[l],3))    # compute the second derivative
            skimage.io.imsave(outpath+'/2rd_hmap'+str(l+1).zfill(3)+fformat,LPH[l],check_contrast=False)
        
        if ed_hm==True:
            edH.append(erode_dilate(H[l],(l+1)*2+1,'square'))     # erode and dilate height map to eliminate boundary spikes, kernel based on layer
            # edH[l]=scipy.ndimage.gaussian_filter(edH[l],3)
            skimage.io.imsave(outpath+'/ed_hmap'+str(l+1).zfill(3)+fformat,edH[l]+1,check_contrast=False)   # add 1 to reflect actual index
            np.savetxt(outpath+'/ed_hmap'+str(l+1).zfill(3)+'.txt',edH[l]+1,fmt='%i')
            if rd_hm==True:
                LPedH.append(get_gaussian_laplace(edH[l],3))    # compute the second derivative
                skimage.io.imsave(outpath+'/2rd_ed_hmap'+str(l+1).zfill(3)+fformat,LPedH[l],check_contrast=False)
            
            if flp_hm==True:
                if (f%2)==1:    # flip even folder temporarily for processing
                    H[l]=numI-1-H[l]
                    edH[l]=numI-1-edH[l]
    
    O=[]
    edO=[]   # initialize a pyramid for refocused stack
    for l in tqdm.tqdm(range(level),desc='  generating refocused images...',position=0,leave=True):     # create a refocused image for each window size
        O.append(get_stack(I,H[l],l+1))     # get original refocused stack
        skimage.io.imsave(outpath+'/refocus'+str(l+1).zfill(3)+fformat,O[l])
        if ed_hm==True:
            edO.append(get_stack(I,edH[l],l+1))     # get modified refocused stack
            skimage.io.imsave(outpath+'/ed_refocus'+str(l+1).zfill(3)+fformat,edO[l],check_contrast=False)
    
    # skimage.io.imsave(rootpath+'/'+str(f+1).zfill(3)+'_GT.tif',O[0],check_contrast=False)
    GT.append(O[0])
    # skimage.io.imsave(rootpath+'/'+str(f+1).zfill(3)+'_GTH.tif',H[0]+1,check_contrast=False)
    GTH.append(H[0]+1)
    # skimage.io.imsave(rootpath+'/'+str(f+1).zfill(3)+'_T.tif',edO[3],check_contrast=False)
    T.append(edO[4])
    # skimage.io.imsave(rootpath+'/'+str(f+1).zfill(3)+'_TH.tif',edH[3]+1,check_contrast=False)
    TH.append(edH[4]+1)
    
    # plt.ioff()  # turn off interactive mode
    # for l in tqdm.tqdm(range(level),desc='  generating surface...',position=0,leave=True):    # generate surface based on height map
    #     surname=rootpath+'/image'+str(f+1).zfill(3)+'_surface'+str(l+1).zfill(3)+fformat
    #     save_surface(H[l],numI,surname)     # save original surface
    #     if ed_hm==True:
    #         surname=rootpath+'/ed_image'+str(f+1).zfill(3)+'_surface'+str(l+1).zfill(3)+fformat
    #         save_surface(edH[l],numI,surname)   # save modificed surface
    
    # output height map total range in txt
    tGT=open(outpath+'/GT.txt','w')  # text file for H
    tGT.write('bottom top \n')
    tT=open(outpath+'/T.txt','w')  # text file for edH
    tT.write('bottom top \n')
    for l in range(level):
        tGT.write(str(H[l].min())+' '+str(H[l].max())+'\n')
        tT.write(str(edH[l].min())+' '+str(edH[l].max())+'\n')
    tGT.close()
    tT.close()
    
    # always correct for height map inversion caused by object translation direction difference (top to bottom OR bottom to top)
    for l in range(level):
        H[l]=numI-H[l]-1  # invert height map
        edH[l]=numI-edH[l]-1
        skimage.io.imsave(outpath+'/inv_hmap'+str(l+1).zfill(3)+fformat,H[l]+1,check_contrast=False)   # add 1 to reflect actual index
        np.savetxt(outpath+'/inv_hmap'+str(l+1).zfill(3)+'.txt',H[l]+1,fmt='%i')
        skimage.io.imsave(outpath+'/inv_ed_hmap'+str(l+1).zfill(3)+fformat,edH[l]+1,check_contrast=False)   # add 1 to reflect actual index
        np.savetxt(outpath+'/inv_ed_hmap'+str(l+1).zfill(3)+'.txt',edH[l]+1,fmt='%i')
    
    if plt_hm==True:
        plt.ion()   # turn on interactive mode
        plot_surface(H[hm_id],1,numI)
        if ed_hm==True:
            plot_surface(edH[hm_id],2,numI)
        
# M=scipy.io.loadmat(matFile)
# shift=M['shift']
# for d in range(len(GT)):
    # i=np.uint8(np.floor(d/2))
    # if d%2==0:
        # skimage.io.imsave(rootpath+'/GT_'+str(i+1).zfill(3)+'.tif',GT[d],check_contrast=False)
        # tmp=np.uint16((GTH[d]+GTH[d+1])/2*0.182-shift[0][i])
        # skimage.io.imsave(rootpath+'/GTH_'+str(i+1).zfill(3)+'.tif',tmp,check_contrast=False)
        # skimage.io.imsave(rootpath+'/T_'+str(i+1).zfill(3)+'.tif',T[d],check_contrast=False)
        # tmp=np.uint16((TH[d]+TH[d+1])/2*0.182-shift[0][i])
        # skimage.io.imsave(rootpath+'/TH_'+str(i+1).zfill(3)+'.tif',tmp,check_contrast=False)
        
            
t_end=time.time()   # end timer
print('\ntime to process: '+str(round(t_end-t_start,3))+'s\n')