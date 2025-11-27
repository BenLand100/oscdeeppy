#  Copyright 2024 by Benjamin J. Land (a.k.a. BenLand100)
#
#  This file is part of OSCDeepPy.
#
#  OSCDeepPy is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  OSCDeepPy is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with OSCDeepPy.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from scipy.optimize import minimize

def transform_coords(xy,dx,dy,theta):
    tx = np.asarray([dx,dy])
    rot = np.asarray([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    return (rot @ xy.T).T + tx

def optimal_transform_fn(ref,other,**kwargs):
    def fn(x):
        tx = transform_coords(ref,*x,**kwargs)
        return np.mean(np.sqrt(np.sum(np.square(other-tx),axis=1)))
    return fn
    
def image_transform_chunked(img, tx_fit, span=100, drizzle=False):
    '''
    tx_fit describes a coordinate transformation that maps a reference coordinate space 
    onto the pixel coordinate space of another image, offset by position and rotation.
    This transformation is used to efficiently and quickly produce a bilinear interpolation
    of img in the reference coordinate space.
    '''

    if drizzle:
        if tx_fit is None:
            tx_fit = type('FakeFitResult',(object,),{'x':(0,0,0)})
        output_shape = (img.shape[0]*2,img.shape[1]*2,img.shape[2])
        y,x = np.arange(output_shape[0])/2,np.arange(output_shape[1])/2
    else:    
        if tx_fit is None:
            return img
        output_shape = img.shape
        y,x = np.arange(img.shape[0]),np.arange(img.shape[1])
        
    # these might not be valid image locations, so keep track of that
    invalid_mask = lambda idx: np.any(idx < 0,axis=1) | (idx[:,0] >= img.shape[1]) | (idx[:,1] >= img.shape[0])

    rgb_out = np.empty(output_shape,dtype=img.dtype)

    for x_i in range(0,len(x),span):
        for y_i in range(0,len(y),span):
            x_span = span if x_i + span <= len(x) else len(x)-x_i
            y_span = span if y_i + span <= len(y) else len(y)-y_i
            chunk_shape = (y_span,x_span,3)
                
            X,Y = np.meshgrid(x[x_i:x_i+x_span],y[y_i:y_i+y_span])
            XY = np.asarray([X.ravel(),Y.ravel()]).T
            XY_tx = transform_coords(XY,*tx_fit.x) # where the reference pixels are in the image space
            XY,X,Y = None,None,None
            
            frac,base = np.modf(XY_tx)  
            # base is the pixel that is "top left" in the frame of the image from the reference coordinates
            # frac is how far fractionally along we are to the next pixel in x or y
            rfrac = frac[:,0]
            lfrac = 1.0-rfrac
            bfrac = frac[:,1]
            tfrac = 1.0-bfrac
            frac = None
            
            # get the (top|bottom), (left|right) corner coordinates for interpolation
            base = np.asarray(base, dtype=np.int16)
            px_invalid = np.zeros(len(XY_tx), dtype=bool)
            
            idx_corner = base.copy()
            px_invalid |= invalid_mask(idx_corner)
            idx_corner[px_invalid] = [0,0]
            rgb_out[y_i:y_i+y_span,x_i:x_i+x_span]  = (tfrac*lfrac*img[idx_corner[:,1],idx_corner[:,0]].T).T.reshape(chunk_shape)
            
            idx_corner = base + [0,1]
            px_invalid |= invalid_mask(idx_corner)
            idx_corner[px_invalid] = [0,0]
            rgb_out[y_i:y_i+y_span,x_i:x_i+x_span] += (bfrac*lfrac*img[idx_corner[:,1],idx_corner[:,0]].T).T.reshape(chunk_shape)
            
            idx_corner = base + [1,0]
            px_invalid |= invalid_mask(idx_corner)
            idx_corner[px_invalid] = [0,0]
            rgb_out[y_i:y_i+y_span,x_i:x_i+x_span] += (tfrac*rfrac*img[idx_corner[:,1],idx_corner[:,0]].T).T.reshape(chunk_shape)
            
            idx_corner = base + [1,1]
            px_invalid |= invalid_mask(idx_corner)
            idx_corner[px_invalid] = [0,0]
            rgb_out[y_i:y_i+y_span,x_i:x_i+x_span] += (bfrac*rfrac*img[idx_corner[:,1],idx_corner[:,0]].T).T.reshape(chunk_shape)
        
            # set invalid pixels to nan
            rgb_out[y_i:y_i+y_span,x_i:x_i+x_span][px_invalid.reshape(chunk_shape[:2])] = np.nan
    
    return rgb_out.reshape(output_shape)
    
    
def find_transformation(ref_pts, img_pts, guess_rot=None, guess_tx=None, verbose=False, max_iter=10, method='Powell', **kwargs):

    star_mask = np.ones(len(ref_pts),dtype=bool)
    for j in range(max_iter):

        pts_ref,pts_other = ref_pts[star_mask],img_pts[star_mask]
        if len(pts_ref) < 10:
            print('Not enough stars')
            break
        guess = [0,0,0]
        if guess_tx is not None:
            guess[0] = guess_tx[0]
            guess[1] = guess_tx[1]
        if guess_rot is not None:
            guess[2] = guess_rot
        tx_fit = minimize(
            optimal_transform_fn(pts_ref,pts_other), 
            guess, method=method, **kwargs
        )
        pts_ref_tx = transform_coords(pts_ref,*tx_fit.x)
        star_dists = np.sqrt(np.sum(np.square(pts_ref_tx - pts_other),axis=1))
        
        good_stars = star_dists < 1.5
        total_good = np.count_nonzero(good_stars)
        marginal_stars = (star_dists < 100.0) & ~good_stars
        total_marginal = np.count_nonzero(marginal_stars)
        bad_stars = ~(good_stars | marginal_stars)
        total_bad = len(pts_ref) - total_good - total_marginal

        if verbose:
            print(f'Iter {j} Good: {total_good} Marginal: {total_marginal} Bad {total_bad}')
    
        if total_marginal == 0 and total_bad == 0:
            if verbose:
                print(f'Good fit; mean distance {np.mean(star_dists)}')
            break
        elif total_bad == 0:
            if verbose:
                print(f'Marginal fit; mean distance {np.mean(star_dists)}')
            star_mask[star_mask] = star_mask[star_mask] & (good_stars)
        else:
            if verbose:
                print(f'Terrible fit; mean distance {np.mean(star_dists)}')
            bad_stars = bad_stars & np.random.randint(2,dtype=bool,size=len(bad_stars))
            star_mask[star_mask] = star_mask[star_mask] & ~bad_stars

    tx_fit.average_distance = np.mean(star_dists)
    tx_fit.total_stars = np.count_nonzero(star_mask)
    tx_fit.star_mask = star_mask
    return tx_fit
