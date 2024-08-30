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

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def show_hist(img, bins=100, log=True):
    '''Display a histogram for the RGB image'''
    plt.hist(img[...,0].ravel(),bins=bins, color='r', histtype='step')
    plt.hist(img[...,1].ravel(),bins=bins, color='g', histtype='step')
    plt.hist(img[...,2].ravel(),bins=bins, color='b', histtype='step')
    if log:
        plt.yscale('log')
    
def display_rgb(img, scale='norm', clip_sigmas=None, plot=True):
    '''Several display options for intermediate RGB image data.'''
    if clip_sigmas is not None:
        mean = np.mean(np.mean(img,axis=0),axis=0)
        std = np.std(np.std(img,axis=0),axis=0)
        img = np.clip(img, mean-std*clip_sigmas, mean+std*clip_sigmas)
    if scale is None:
        pass
    elif scale == 'norm':
        levels = np.max(np.max(img,axis=0),axis=0)
        img = img / levels
    elif scale == 'linear':
        levels = np.max(np.max(img,axis=0),axis=0)
        floors = np.min(np.min(img,axis=0),axis=0)
        img = (img - floors) / (levels-floors)
    elif scale == 'auto':
        if clip_sigmas is None:
            mean = np.mean(np.mean(img,axis=0),axis=0)
            std = np.std(np.std(img,axis=0),axis=0)
        img = (img - mean) / (std * 2.5) + 0.5
    img = np.clip(img,0,1)
    if plot:
        plt.imshow(img)
    else:
        return Image.fromarray(np.asarray(255*img,dtype=np.uint8))

def linear_rgb(img, lower_limit=0, upper_limit=1, figsize=[14,7], save=None, plot=True, **kwargs):
    '''
    Display and save linear [0,1] RGB data, with simple linear stretching 
    optional to support other ranges.
    '''
    plt.figure(figsize=figsize)
    if lower_limit != 0 or upper_limit != 1:
        rgb = np.clip(((img-lower_limit)/(upper_limit-lower_limit)),0,1)
    else:
        rgb = img
    pil_img = Image.fromarray(np.asarray(255*rgb,dtype=np.uint8))
    if save:
        pil_img.save(save, **kwargs)
    if plot:
        plt.imshow(rgb)
    else:
        return pil_img
        
def draw_constellations(tri,color='k',**kwargs):
    ax = plt.gca()
    for t in tri:
        ax.add_patch(plt.Polygon(t.abc,facecolor='none',edgecolor=color,**kwargs))


