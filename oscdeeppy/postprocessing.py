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

def equalize_img(img, sigma_cut=3, get_stats=False):
    ''' Balances color channels by shifting them such that the background peak is gray '''
    normd = np.empty_like(img)
    avg_lvl = 0
    avg_spd = 0
    for ch in range(3):
        val = img[...,ch]
        if ch == 1:
            val = val/2
        c = np.median(val)
        s = np.std(val[(val > c*0.8)&(val < c*1.25)])
        avg_lvl += c/3
        avg_spd += s/3
        normd[...,ch] = (val-c)#/s

    normd = avg_lvl+normd#*avg_spd
    m,s = np.median(normd),np.std(normd)
    eqlz = normd - m + sigma_cut*s
    if get_stats:
        return eqlz,sigma_cut*s,s
    return eqlz
    
def set_blackpoint_and_norm(rgb, blackpoint=None, auto_blackpoint=1e-3, whitepoint=None, auto_whitepoint=1-1e-5):
    '''
    Either clips and rescales the image such that the range [blackpoint,whitepoint]
    is represented by the range [0,1]. If blackpoint or whitepoint are not specified,
    they are calculated as the quantiles auto_blackpoint or auto_whitepoint in the
    RGB channel intensities.
    '''
    if blackpoint is None and whitepoint is None:
        blackpoint,whitepoint = np.quantile(rgb,[auto_blackpoint,auto_whitepoint])
    elif whitepoint is None:
        whitepoint = np.quantile(rgb,auto_whitepoint)
    elif blackpoint is None:
        blackpoint = np.quantile(rgb,auto_blackpoint)
    return blackpoint,whitepoint,np.clip((rgb-blackpoint)/(whitepoint-blackpoint),0,1)
    
def hist_bkg_fn(x, mean, sigma, A, B):
    return A+B*np.exp(-np.square((x-mean)/sigma/2))

def hist_bkg_residual(x,y):
    def fn(args):
        y_guess = hist_bkg_fn(x, *args)
        return np.sum(np.square(y-y_guess))
    return fn
    
def histogram_metrics(rgb):
    '''
    Fits a gaussian distribution to the channel intensity distribution to determine
    the properties of the primary component of the image, assumed to be background
    pixels.
    '''
    
    hist_median,hist_sigma = np.median(rgb),np.std(rgb)
    counts,edges = np.histogram(rgb,bins=np.linspace(hist_median-hist_sigma*2,hist_median+hist_sigma*2,500))
    centers = (edges[:-1]+edges[1:])/2
    
    fit = minimize(hist_bkg_residual(centers, counts), (hist_median,hist_sigma,0,np.max(rgb)), method='Nelder-Mead')
    fit.bkg_mean,fit.bkg_sig,fit.bkg_bkg,fit.bkg_peak = fit.x
    fit.hist_median,fit.hist_sigma = hist_median,hist_sigma
    
    return fit
    
    
def ghs(rgb, low_point=0, high_point=1, symmetry_point=0.5, stretch_factor=0, form=1):
    ''' See: https://www.ghsastro.co.uk/doc/tools/GeneralizedHyperbolicStretch/GeneralizedHyperbolicStretch.html '''
    D = np.exp(stretch_factor)-1
    spread = high_point-low_point
    rgb = (rgb-symmetry_point)/spread
    mask = rgb >= 0.0
    result = np.empty_like(rgb)
    if form > 0:
        fn = lambda arg: 1.0 - np.power(1.0 + form*D*arg, -1.0/form)
    elif form == 0:
        fn = lambda arg: 1.0 - np.exp(-D*arg)
    elif form == -1:
        fn = lambda arg: np.log(1.0 + D*arg)
    else:
        fn = lambda arg: (1.0 - np.power(1.0 - form*D*arg,(form+1.0)/form)) / (D*(form+1))
    result[mask] = fn(rgb[mask])
    mask = ~mask
    result[mask] = -fn(-rgb[mask])
    bound = symmetry_point / spread
    a = -fn(bound)
    b = fn(1.0 - bound)
    result = (result-a)/(b-a)
    return np.clip(result,0,1)
    
    
def optimal_ghs_stretch(a,b,**kwargs):
    '''
    Finds the GHS stretch factor that best transforms a into b, given other GHS factors.
    '''
    def fn(x):
        stretch, = x
        b_fit = ghs(a,stretch_factor=stretch,**kwargs)
        return np.sum(np.square(b-b_fit))
    return fn
