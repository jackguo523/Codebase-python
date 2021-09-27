# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:39:12 2020

@author: JACK
"""

# this script stitches MUSE scan tiles with fixed shifts

import argparse
import glob
import numpy as np
import tqdm
import time
import sys
from PIL import Image


# equally blend two images
# input: first image img1, second image img2, result width w, result height h, shift s
# output: blend image b
def blend(img1,img2,w,h,s):
    newimg1=Image.new('RGBA',size=(w,h),color=(0,0,0,0))
    newimg1.paste(img2,s)   # paste img1 on top of img2
    newimg1.paste(img1,(0,0))
    
    newimg2=Image.new('RGBA',size=(w,h),color=(0,0,0,0))
    newimg2.paste(img1,(0,0))   # paste img2 on top of img1
    newimg2.paste(img2,s)
    
    b=Image.blend(newimg1,newimg2,alpha=0.5)   # blend two images with alpha=0.5
    
    return b


# add input arguments and parse it to an object
parser=argparse.ArgumentParser(description='stitch')
parser.add_argument('--filepath',metavar='filepath',help='file root directory to be processed')   # root directory
parser.add_argument('--overlap',type=int,default=5,help='overlap rate')
parser.add_argument('--dimension',type=int,nargs=2,metavar=('row','col'),default=[3,3],help='spatial dimension of the mosaic') # [row,col] in snake scan

# parse and read arguments
args=parser.parse_args()
filepath=args.filepath
overlap=args.overlap
row,col=args.dimension

filemask=filepath+'/*.tif'
files=glob.glob(filemask)
numI=len(files)

xoverlap=np.floor(4096*(100-overlap)/100.0).astype(np.uint16)
yoverlap=np.floor(2160*(100-overlap)/100.0).astype(np.uint16)


t_start=time.time()
row_tiles=[]
for i in tqdm.tqdm(range(row),desc='start row stitching...',position=0,leave=True): # row stitching
    result=Image.open(files[i*col]) # read the first image
    for j in range(1,col):
        ow,oh=result.size # get current width and height
        nw=ow+xoverlap # get new width
        nh=oh # get new height
        tmp=Image.open(files[i*col+j])  # load a new image
        if i%2==0:  # moving right in odd rows
            xshift=j*xoverlap
            result=blend(result,tmp,nw,nh,(xshift,0))
        else:   # moving left in even rows
            xshift=xoverlap
            result=blend(tmp,result,nw,nh,(xshift,0))
    row_tiles.append(result)
    
# for d in range(len(row_tiles)):
#     row_tiles[d].save('{:03d}.tif'.format(d+1))


stitched=row_tiles[0] # load the first row-stitched image
for d in tqdm.tqdm(range(1,row),desc='start column stitching...',position=0,leave=True): # column stitching
    ow,oh=stitched.size # get current width and height
    nw=ow
    nh=oh+yoverlap    
    tmp=row_tiles[d] # load a new row-stitched image
    yshift=d*yoverlap
    stitched=blend(stitched,tmp,nw,nh,(0,yshift))

t_end=time.time() 
sys.stdout.write("\ntime to process: "+str(round(t_end-t_start,3))+"s\n")
    
stitched.save('stitched.tif')



