# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:39:12 2020

@author: JACK
"""


# The MUSE-SCAN program collects multiple frames for each tile for refocusing
# Thie program refocuses each tile using spatial methods


import skimage.io
import skimage.color
import skimage.transform
import skimage.filters
import numpy as np
import os
import argparse
import glob
import time
import sys
import tqdm
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

file_format='*.tif'

def stdout(words):
    sys.stdout.write(str(words)+'\n')
    sys.stdout.flush()

# focus measure metrics
def GLVA(mat):    # graylevel variance
    fm=np.std(mat)
    return fm
def HISE(mat):    # histogram entropy
    fm=skimage.measure.shannon_entropy(mat)
    return fm
def SPFQ(mat):    # spatial frequency
    m=mat.shape[0]
    n=mat.shape[1]
    rf=np.zeros((m,n),np.int8)     # row frequency
    cf=np.zeros((m,n),np.int8)     # column frequency
    rf[0:m-1,:]=mat[1:m,:]-mat[0:m-1,:]
    cf[:,0:n-1]=mat[:,1:n]-mat[:,0:n-1]
    rf=np.mean(np.float_power(rf,2))
    cf=np.mean(np.float_power(cf,2))
    fm=np.sqrt(rf+cf)
    return fm
def BREN(mat):    # first-order differentiation -- Brenner's
    m=mat.shape[0]
    n=mat.shape[1]
    rf=np.zeros((m,n),np.int8)     # row frequency
    cf=np.zeros((m,n),np.int8)     # column frequency
    rf[0:m-2,:]=mat[2:m,:]-mat[0:m-2,:]
    cf[:,0:n-2]=mat[:,2:n]-mat[:,0:n-2]
    fm=np.maximum(rf,cf)
    fm=np.float_power(fm,2)
    fm=np.mean(fm)
    fm=np.sqrt(fm)
    return fm
    
# calculate focus measure of a given region
def fmeasure(mat, metric):
    if metric=='GLVA':
        fm=GLVA(mat)
        return fm
    elif metric=='HISE':
        fm=HISE(mat)
        return fm
    elif metric=='SPFQ':
        fm=SPFQ(mat)
        return fm
    elif metric=='BREN':
        fm=BREN(mat)
        return fm
    else:
        print("invalid focus measure metric")
        quit()
                  

# batch multi-focus fusion
def multi_focus_fusion(idx,rootpath,filepath,metric,block,majority,hmap_flag):
    if not os.path.exists(filepath+'/result'):
        os.mkdir(filepath+'/result')    # create output folder if not exist
    I=[]    # rgb images
    gI=[]   # grayscale images
    filemask=filepath+'/'+file_format  # create file mask
    files=glob.glob(filemask)   # get all file names
    numI=len(files)     # get the file count
    
    for d in tqdm.tqdm(range(numI),desc='Loading images...',position=0,leave=True): # read all files
        I.append(skimage.io.imread(files[d]))
        if I[d].ndim==3:
            tmp_gI=0.2989*I[d][:,:,0]+0.5870*I[d][:,:,1]+0.1140*I[d][:,:,2]     # remember that scikit-image uses different formula
            #tmp_gI=np.uint8(skimage.color.rgb2gray(I[i])*256)
        else:
            tmp_gI=I[d]
        gI.append(np.uint8(np.round(tmp_gI)))     # convert to grayscale for faster focus evaluation

    Ix=gI[0].shape[1]   # get image width
    Iy=gI[0].shape[0]   # get image height
    
    bx=int(Ix/block[1])  # get partitioned block count along x-axis -> col
    by=int(Iy/block[0])  # get partitioned block count along y-axis -> row
    
    # create list for partitioned blocks and corresponding focus measure values
    tmp_blocks=np.zeros((numI*by*bx,)+(block[0],block[1]), np.uint8)
    blocks=np.reshape(tmp_blocks,(numI,by,bx,block[0],block[1]))
    focus_measure=np.zeros((numI,)+(by,bx))

    # partition all images into blocks and calculate the focus measure of each block for evaluation
    for d in tqdm.tqdm(range(numI),desc='Performing partition...',position=0,leave=True):   # for every frame
        counti=0    # row count
        for i in range(0,Iy-block[0]+1,block[0]):   # for every row
            countj=0    # column count
            for j in range(0,Ix-block[1]+1,block[1]):   # for every column
                blocks[d,counti,countj,]=gI[d][i:i+block[0],j:j+block[1]]   # transfer current block
                focus_measure[d,counti,countj]=fmeasure(blocks[d,counti,countj,],metric)   # evaluate the focus measure of current block
                countj+=1
            counti+=1
    
    # get the corresponding maximum focus measure indices and fuse the images
    stdout('\nPerforming fusion...')
    if I[0].ndim==3:
        interI=np.zeros([Iy,Ix,3],np.uint8)     # initialize intermediate image
    else:
        interI=np.zeros([Iy,Ix],np.uint8)       # initialize intermediate image
    height_map=np.zeros((by,bx),np.uint16)      # initialize intermediate height map
    height_map_full=np.zeros((Iy,Ix),np.uint8)  # height map in high resolution
    counti=0
    for i in range(0,Iy-block[0]+1,block[0]):
        countj=0
        for j in range(0,Ix-block[1]+1,block[1]):
            tmp=np.zeros(numI)
            for d in range(numI):
                tmp[d]=focus_measure[d,counti,countj]   # organize focus measure values from all frames
            max_index=np.uint16(np.argmax(tmp))         # get the index corresponding to the maximum focus measure
            if I[0].ndim==3:
                interI[i:i+block[0],j:j+block[1],:]=I[max_index][i:i+block[0],j:j+block[1],:]   # fuse current block
            else:
                interI[i:i+block[0],j:j+block[1]]=I[max_index][i:i+block[0],j:j+block[1]]   # fuse current block
            height_map[counti,countj]=max_index     # save the index
            countj+=1
        counti+=1
    
    skimage.io.imsave(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-inter-fuse.tif',interI,check_contrast=False)   
    np.savetxt(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-inter-hmap.txt',height_map+1,fmt='%d')
    height_map_img=np.uint8(height_map/np.max(height_map)*255)
    for i in range(by):
        for j in range(bx):
            height_map_full[i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1]]=height_map_img[i,j]
    skimage.io.imsave(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-inter-hmap.tif',height_map_full,check_contrast=False)
    
    # apply a majority filter
    #print('Finalizing fusion...')
    if I[0].ndim==3:
        finalI=np.zeros([Iy,Ix,3],np.uint8)
    else:
        finalI=np.zeros([Iy,Ix],np.uint8)
    selem=np.ones((majority,majority),bool)
    new_height_map=np.zeros((by,bx),np.uint16)       # initialize final height map
    new_height_map_full=np.zeros((Iy,Ix),np.uint8)   # height map in high resolution
    skimage.filters.rank.majority(height_map,selem,out=new_height_map)  # apply a majority filter with fixed size to reduce bubble artifacts
    for i in range(by):
        for j in range(bx):
            if I[0].ndim==3:
                finalI[i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1],:]=I[new_height_map[i,j]][i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1],:]     # fuse current block to the final image
            else:
                finalI[i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1]]=I[new_height_map[i,j]][i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1]]     # fuse current block to the final image
    skimage.io.imsave(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-fuse.tif',finalI,check_contrast=False)   
    skimage.io.imsave(rootpath+'/{:03d}.tif'.format(idx+1),finalI,check_contrast=False) # save the result in root folder for stitching
    np.savetxt(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-hmap.txt',new_height_map+1,fmt='%d')
    new_height_map_img=np.uint8(new_height_map/np.max(new_height_map)*255)
    for i in range(by):
        for j in range(bx):
            new_height_map_full[i*block[0]:(i+1)*block[0],j*block[1]:(j+1)*block[1]]=new_height_map_img[i,j]
    skimage.io.imsave(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-hmap.tif',new_height_map_full,check_contrast=False)

    if hmap_flag:       # plot the height map if required -- can be 3D mesh texture mapped with the original image
        fig=plt.figure()
        ax=Axes3D(fig)
        X=np.arange(0,bx,1)
        Y=np.arange(0,by,1)
        X,Y=np.meshgrid(X,Y)
        surf=ax.plot_surface(Y,X,new_height_map+1,rstride=1,cstride=1,linewidth=0,antialiased=True,cmap=cm.coolwarm)
        fig.colorbar(surf,shrink=0.3,aspect=20)
        ax.set_xlim(0,by)
        ax.set_ylim(0,bx)
        ax.set_zlim(1,numI)
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        ax.view_init(elev=90,azim=0)
        #plt.imshow(bI,zorder=0)
        #bI=plt.imread(filepath+'/result/'+str(block[0])+'x'+str(block[1])+'-'+str(metric)+'-fuse.jpg')
        #bI=skimage.transform.downscale_local_mean(finalI[:,:,],(27,32))
        #bI=skimage.transform.resize(finalI,(by,bx))
        #ax.plot_surface(Y,X,np.atleast_2d(1),rstride=10,cstride=10,facecolors=bI)
        plt.show()


# add input arguments and parse it to an object
parser=argparse.ArgumentParser(description='Batch multi-focus fusion')
parser.add_argument('--filepath',metavar='filepath',help='file root directory to be processed')   # root directory
#parser.add_argument('--filename',metavar='filename',help='template filename to be processed')
#parser.add_argument('--dim',type=int,nargs=2,default=[1, 1],metavar=('X', 'Y'),help='mosaic lateral dimensions')
parser.add_argument('--metric',metavar='metric',help='focus measure metric to be applied')     # user selected focus measure metric
parser.add_argument('--block',type=int,nargs=2,default=[10, 10],metavar=('X', 'Y'),help='partitioned block size')  # focus measure block size
parser.add_argument('--majority',type=int,default=1,help='majority filter range')    # majority filter disk size
parser.add_argument('-l','--hmap',action='store_true',help='flag for height map')    # flag evoking height map when set

# parse and read arguments
args=parser.parse_args()
root_filepath=args.filepath
#template=args.filename
#dimension=args.dim
metric=args.metric
block=args.block    # [0]->row, [1]->col
majority=args.majority
if args.hmap is not False:  # check if --hmap is set, if set -> set hmap_flag to True
    hmap_flag=True
else:
    hmap_flag=False

# folders=os.listdir(root_filepath)    # check folders in input directory
(_,folders,files) = next(os.walk(root_filepath))
count=len(folders)   # get folder counts

t_start=time.time()
stdout('\n*****START MULTI-FOCUS FUSION*****\n')
for i in range(count):
    stdout('Start batch #'+str(i+1)+'...')
    current_filepath=root_filepath+'/'+str(folders[i])
    multi_focus_fusion(i,root_filepath,current_filepath,metric,block,majority,hmap_flag)
t_end=time.time()
sys.stdout.write("\nTime to process: "+str(round(t_end-t_start,3))+"s\n")
