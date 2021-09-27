# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:39:48 2021

@author: Jack
"""

# this script is developed for muse deconvolution

import skimage.io
import os
import numpy as np
import numpy.random as npr
from scipy.signal import fftconvolve, convolve
import warnings
import cv2

def blind_richardson_lucy(image, psf=None, iterations=10, return_iterations=False, clip=False):
    """Blind Richardson-Lucy deconvolution.
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray, optional
       A first estimate of the point spread function, same size as image
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation. After a given iterations, the estimates can produce
       division by 0 problems, then the algorithm is automatically stopped.
    return_iterations : boolean, optional
        Returns instead of a tuple of the final restorated image and used PSF
        a tuple of all iterations for further investigation
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.
    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    psf : ndarray
        The last PSF estimate to deconvolve image
    Examples
    --------
    >>> from skimage.restoration import blind_richardson_lucy
    >>> image = np.zeros((100,100))
    >>> im[40:60, 45:55] = 1
    >>> im[45:55, 40:60] = 1
    >>> psf = np.zeros_like(image)
    >>> psf[50,50] = 1
    >>> psf = gaussian(psf, 2)
    >>> image_conv = convolve2d(image, psf, 'same')
    >>> deconvolved, calc_psf = blind_richardson_lucy(image_conv, 10)
    Notes
    -----
    This function estimates a point spread function based on an inverse Richardson Lucy algorithm as described
    in Fish et al., 1995. It is an iterative process where the PSF and image is deconvolved, respectively.
    It is more noise tolerant than other algorithms, such as Ayers-Dainty and the Weiner filter algorithms (taken
    from the paper).
    The algorithm performs well with gaussian PSFs and can recover them nicely without any prior knowledge. If
    one has already an educated guess, one should pass the PSF as argument to the function.
    Note, that the PSF should have the same shape as the image, and the PSF should be centered.
    Due to its nature, the algorithm may divide by 0. The function catches this issue and aborts the iterative
    process. Mostly, the optimal number of iterations is before this error may occur.
    References
    ----------
    .. [1] Fish, D. A., A. M. Brinicombe, E. R. Pike, and J. G. Walker.
           "Blind deconvolution by means of the Richardson–Lucy algorithm."
           JOSA A 12, no. 1 (1995): 58-65.
           https://pdfs.semanticscholar.org/9e3f/a71e22caf358dbe873e9649f08c205d0c0c0.pdf
    """
    if return_iterations:
        all_iterations = np.empty((iterations, 2,) + image.shape)

    # Convert image to float for computations
    image = image.astype(np.float)

    # Initialize im_deconv and PSF
    im_deconv = np.full(image.shape, 0.5)

    if psf is None:
        psf = np.full(image.shape, 0.5)
    else:
        assert psf.shape == image.shape, 'Image and PSF should have the same shape!'
        psf = psf.astype(np.float)

    for i in range(iterations):
        # Deconvolve the PSF
        # Hack: in original publication one would have used `image`,
        #       however, this does not work. Using `im_deconv` instead recovers PSF.
        relative_blur_psf = im_deconv / fftconvolve(psf, im_deconv, 'same')

        # Check if relative_blur_psf contains nan, causing the algorithm to fail
        if np.count_nonzero(~np.isnan(relative_blur_psf)) < relative_blur_psf.size:
            warnings.warn('Iterations stopped after {} iterations because PSF contains zeros!'.format(i),
                          RuntimeWarning)
            break

        else:
            psf *= fftconvolve(relative_blur_psf, im_deconv[::-1, ::-1], 'same')

            # Compute inverse again
            psf_mirror = psf[::-1, ::-1]

            # Standard Richardson-Lucy deconvolution
            relative_blur = image / fftconvolve(im_deconv, psf, 'same')
            im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

            # Add iteration to list, if desired
            if return_iterations:
                all_iterations[i, 0] = im_deconv.copy()
                all_iterations[i, 1] = psf.copy()

    # Don't know if this makes sense here...
    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    if return_iterations:
        return all_iterations

    else:
        return im_deconv, psf


path='test/001.tif'

# (_,_,files)=next(os.walk(path))
# files=[i for i in files if 'tif' in i.lower()]

# I=[]
# for f in range(len(files)):
#     tmp=skimage.io.imread(path+'/'+files[f])
#     I.append(tmp)
I=skimage.io.imread(path)
grayI=np.uint8(np.round(0.2989*I[:,:,0]+0.5870*I[:,:,1]+0.1140*I[:,:,2]))

dI,psf=blind_richardson_lucy(grayI,psf=None,iterations=20)

skimage.io.imsave('test/002-decon.tif',cv2.normalize(dI,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC1))