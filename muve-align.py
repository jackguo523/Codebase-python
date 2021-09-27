import skimage.io
import skimage.color
import numpy
import cv2
import tqdm
import scipy.ndimage
# import sys
import os
import glob
# import imagestack

#get the transform that aligns image B to image A
def align(A, B, max_power=5):       #### try to play with this number a bit, ususally larger offset requires larger max_power
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
             
    # Specify the number of iterations.
    number_of_iterations = 5000
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    #attempt to fit the image, increasing the blur as necessary
    i = 3
    while i <= max_power:    
        #blur the image used for fitting
        im1_blur = scipy.ndimage.filters.gaussian_filter(A, (2 ** i, 2 ** i), mode='reflect')
        im2_blur = scipy.ndimage.filters.gaussian_filter(B, (2 ** i, 2 ** i), mode='reflect')
         
        # Run the ECC algorithm. The results are stored in warp_matrix.
        # Define 2x3 matrix and initialize the matrix to identity
        warp_matrix = numpy.eye(2, 3, dtype=numpy.float32)
        try:
            (cc, warp_matrix) = cv2.findTransformECC (im1_blur,im2_blur,warp_matrix, warp_mode, criteria, None, 3)
        except:
            #print("Error aligning at p = " + str(i))
            i = i + 1
        else:
            #print("Successful alignment at p = " + str(i))
            break
    #enforce the fact that the x-axis is already aligned
    #warp_matrix[0, 2] = 0
    return warp_matrix
    
    #if i > 0:
    #    (cc, warp_matrix) = cv2.findTransformECC (A,B,warp_matrix, warp_mode, criteria)
        #warp_matrix[0, 2] = 0
    #    return warp_matrix

filemask = "C:/Users/jack/Desktop/test/001/*.tif"     #### your input files - images
out_dir = "C:/Users/jack/Desktop/test/001/aligned"	  #### output directory

files=glob.glob(filemask)   # get all file names
numI=len(files)     # get the file count

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

warps = [numpy.eye(2, 3, dtype=numpy.float32)]
for d in tqdm.tqdm(range(numI),desc='Calculating and applying alignment transformation...',position=0,leave=True):
    if d!=0:
        S1=skimage.io.imread(files[d-1])
        S2=skimage.io.imread(files[d])
        G1=skimage.color.rgb2gray(S1.astype(numpy.float32))
        G2=skimage.color.rgb2gray(S2.astype(numpy.float32))
        warps.append(align(G1,G2))
        warps[d][0, 2] = warps[d][0, 2] + warps[d-1][0, 2]
        warps[d][1, 2] = warps[d][1, 2] + warps[d-1][1, 2]
    
    S=skimage.io.imread(files[d])
    
    I=cv2.warpAffine(S,warps[d],(S.shape[1],S.shape[0]),flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    skimage.io.imsave(out_dir+'/{:03d}.tif'.format(d+1),I,check_contrast=False)