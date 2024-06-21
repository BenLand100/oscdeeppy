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

def debayer_simple(filter_img):
    '''
    The 'super pixel' algorithm that halves the image resolution
    '''
    R = filter_img[0::2,0::2]
    G1 = filter_img[0::2,1::2]
    G2 = filter_img[1::2,0::2]
    G = (G1+G2)/2
    B = filter_img[1::2,1::2]
    return np.asarray([R,G,B]).transpose(1,2,0)

def debayer_full(filter_img, dtype=None):
    '''
    Linearly interopolated color channels averaging: the nearest two pixels of 
    the appropriate channel when adjcent to the desired color, the nearest four 
    pixels when diagonal, and keeping the pixel color when it is of the right
    channel. TODO: is this sane?
    '''
    if dtype == None:
        dtype = filter_img.dtype
    R = filter_img[0::2,0::2]
    G1 = filter_img[0::2,1::2]
    G2 = filter_img[1::2,0::2]
    B = filter_img[1::2,1::2]
    deb = np.zeros(filter_img.shape+(3,), dtype=dtype)
    
    deb[0::2,0::2,0] = R
    deb[0::2,0::2,1] = 0.25*(G1+G2)
    deb[0::2,2::2,1] += 0.25*(G1[:,:-1])
    deb[2::2,0::2,1] += 0.25*(G2[:-1,:])
    deb[0::2,0::2,2] = 0.25*B
    deb[0::2,2::2,2] += 0.25*B[:,:-1]
    deb[2::2,0::2,2] += 0.25*B[:-1,:]
    deb[2::2,2::2,2] += 0.25*B[:-1,:-1]
    
    deb[0::2,1::2,0] = 0.5*R
    deb[0::2,1:-2:2,0] += 0.5*R[:,1:]
    deb[0::2,1::2,1] = G1
    deb[0::2,1::2,2] = 0.5*B
    deb[2::2,1::2,2] += 0.5*B[:-1,:]
    
    deb[1::2,0::2,0] = 0.5*R
    deb[1:-2:2,0::2,0] += 0.5*R[1:,:]
    deb[1::2,0::2,1] = G2
    deb[1::2,0::2,2] = 0.5*B
    deb[1::2,2::2,2] += 0.5*B[:,:-1]
    
    deb[1::2,1::2,0] = 0.25*R
    deb[1::2,1:-2:2,0] += 0.25*R[:,1:]
    deb[1:-2:2,1::2,0] += 0.25*R[1:,:]
    deb[1:-2:2,1:-2:2,0] += 0.25*R[1:,1:]
    deb[1::2,1::2,1] = 0.25*(G1+G2)
    deb[1:-2:2,1::2,1] += 0.25*(G1[1:,:])
    deb[1::2,1:-2:2,1] += 0.25*(G1[:,1:])
    deb[1::2,1::2,2] = B
    return deb
