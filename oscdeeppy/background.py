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
from scipy.spatial.distance import cdist

def huber(f, delta):
    '''
    Huber loss, which is quadratic while f is less than delta, and linear above.
    This form reduces the impact of outliers substantially while leaving data 
    consistent with the model in the least-squares paradigm.'''
    abs_f = np.abs(f)
    mask = abs_f > delta
    res = np.empty_like(f)
    res[mask] = 2.0*delta*(abs_f[mask] - 0.5*delta)
    mask = ~mask
    res[mask] = np.square(f[mask])
    return res

class Poly2D:
    '''
    A coefficient representation of a 2D polynomial surface made of terms like
    coeff_ij * x^i * y^j with i,j from 0 to order.
    '''
    def __init__(self, order=2, coeff=None):
        '''
        Initialize the polynomial of order given by order to f=0 unless coeff is
        specified, then the coefficients from coeff are used to represent an
        arbitrary polynomial.
        '''
        self.coeff_shape = (order+1,order+1)
        self.coeff = np.zeros(self.coeff_shape) if coeff is None else coeff
        i = np.arange(order+1)
        I,J = np.meshgrid(i,i)
        self.coeff_mask = I+J <= order
        self.order = order
        
    def fit(self, x, y, z, **kwargs):
        '''
        Fit the polynomial to a set of points to attempt to satisfy f(x,y) = z 
        using Huber loss and the Powell method.
        '''
        def fn(_x):
            coeff = np.empty(self.coeff_shape)
            coeff[self.coeff_mask] = _x
            z_poly = Poly2D._eval(x, y, self.order, coeff)
            residual = z - z_poly
            return np.sum(huber(residual,5))
        fit = minimize(fn, self.coeff[self.coeff_mask], method='Powell', tol=1e-4, **kwargs)
        #print(fit)
        self.coeff[self.coeff_mask] = fit.x
    
    def __call__(self, x, y):
        return Poly2D._eval(x, y, self.order, self.coeff)
    
    def _eval(x, y, order, coeff):
        assert x.shape == y.shape
        result = np.zeros(x.shape)
        for i in range(order+1):
            for j in range(order+1):
                if j + i > order:
                    continue
                result += coeff[i,j] * x**i * y**j
        return result
        

class ThinPlateSpline2D:
    '''
    A [thin plate spline](https://en.wikipedia.org/wiki/Thin_plate_spline) 
    is an optimal way of interpolating data. This is specialized to f(x,y) = z 
    for the purpose of fitting background gradients, and is implemented using
    radial basis functions of the form log(r)r^2.
    '''
   
    def __init__(self, ctrl_x=None, ctrl_y=None, values=None, params=None, **kwargs):
        if values is None:
            if params is None: # Empty init
                assert ctrl_x is None, 'ctrl_x argument ignored as no values specified'
                assert ctrl_y is None, 'ctrl_y argument ignored as no values specified'
                self.ctrl_x = None
                self.ctrl_y = None
                self.params = None
            else: # pre-fit TPS
                assert ctrl_x is not None, 'ctrl_x argument required when params specified'
                assert ctrl_y is not None, 'ctrl_y argument required when params specified'
                self.ctrl_x = ctrl_x
                self.ctrl_y = ctrl_y
                self.params = params
        else: # expecting a fit
            assert params is None, 'params argument ignored because values specified'
            self.fit(ctrl_x, ctrl_y, values, **kwargs)        
        
    def fit(self, x, y, z, smoothing=0.0):
        '''
        Find the thin plate spline that satisfies f(x,y) = z with an optional
        smoothing regularization.
        '''
        self.ctrl_x = np.asarray(x).ravel()
        self.ctrl_y = np.asarray(y).ravel()
        self.ctrl_pts = np.asarray([self.ctrl_x,self.ctrl_y]).T
        
        X = np.asarray([self.ctrl_x,self.ctrl_y]).T # (xi,yi) stacked as rows
        P = np.hstack([np.ones((len(X), 1)), X]) # (1,xi,yi) stacked rows for poly terms
        R = self._rbf_mat(x,y) + smoothing * np.identity(len(X)) # RBF matrix + regularization

        A = np.vstack([
            np.hstack([R, P]), 
            np.hstack([P.T, np.zeros((3,3))])
        ])

        Y = np.vstack([z.reshape((len(X),1)), np.zeros((3, 1))])

        #Solves A*params = Y
        self.params = np.linalg.solve(A, Y)
        
        return self.params
    
    def _rbf(self, r, epsilon=1e-32):
        result = np.empty_like(r)
        mask = r > epsilon
        r_valid = r[mask]
        result[mask] = np.square(r_valid)*np.log(r_valid)
        result[~mask] = 0.0
        return result
    
    def _rbf_mat(self, x, y):
        return self._rbf(cdist(np.asarray([x,y]).T,self.ctrl_pts))
    
    def _eval(self, _x, _y):
        x,y = _x.ravel(), _y.ravel()
        weights = self.params[:-3]
        offset,linx,liny = self.params[-3:]
        result = (self._rbf_mat(x,y) @ weights).squeeze() + x*linx + y*liny + offset
        return result.reshape(_x.shape)
    
    def __call__(self, x, y):
        return self._eval(x, y)
    
def fit_background_poly(channel, sample_frac=1e-4, sigma_cut=3, order=3):
    '''
    Given an RGB image channel (a grayscale image) sample sample_frac points 
    with a value within sigma_cut sigma (calculated as the standard deviation of
    the image) of the median value of the image. Use these points to fit a 
    Poly2D of the specified order.
    '''
    mean = np.median(channel)
    std = np.std(channel)
    mask = np.abs(channel-mean) <= std*sigma_cut
    valid_pts = np.argwhere(mask)
    pass_frac = len(valid_pts)/len(channel.ravel())
    sample_frac = sample_frac / pass_frac
    assert sample_frac <= 1.0, 'Too few remaining points'
    idx = np.random.choice(np.arange(len(valid_pts)), size=int(len(valid_pts)*sample_frac))
    pts = valid_pts[idx]
    y, x = pts[:,0],pts[:,1]
    
    p = Poly2D(order=order)
    p.fit(x,y,channel[y,x])
    return x, y, p

def polyfit_bkg_extract(img, return_bkg=False, order=3, sigma_cut=3, **kwargs):
    '''
    Applies fit_background_poly to all three channels of img, and subtracts the
    resulting polynomial values from the image channels. 
    To avoid under/overshoot or imbalance, color channels are then shifted such
    that the overall median level is sigma_cut sigmas (calculated as the standard 
    deviation of the image) above zero.
    '''
    
    bkg = np.empty_like(img)
    for ch in range(3):
        pts_x,pts_y,poly = fit_background_poly(img[...,ch], order=order, sigma_cut=sigma_cut, **kwargs)
    
        y = np.arange(img.shape[0])
        x = np.arange(img.shape[1])
        X,Y = np.meshgrid(x,y)
        bkg[...,ch] = poly(X,Y)
    
    img_bkg = img - bkg
    m,s = np.median(img_bkg),np.std(img_bkg)
    img_bkg = img_bkg - m + sigma_cut*s
    
    if return_bkg:
        return img_bkg, bkg
    return img_bkg
    
