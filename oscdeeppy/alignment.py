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

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from .transform import transform_coords
    
class TriHash:
    '''This represents a geometrically hashable triangle, such that triangles of 
       with the same side lengths when sorted are equal. Also sortable by area.
       
       Keeps track of the original edge points so coordinate system comparisons
       are possible when similar triangles are identified.'''
    
    def __init__(self, a, b, c, ref=None):
        ab = a-b
        bc = b-c
        ca = c-a
        dists = np.sqrt(np.asarray([np.dot(ab,ab), np.dot(bc,bc), np.dot(ca,ca)]))
        cross_pts = np.asarray([c,a,b])
        idx = np.argsort(-dists)
        self.dists = dists[idx]
        self.abc = cross_pts[idx]
        self.ref = ref
        s = 0.5*np.sum(self.dists)
        self.area = np.sqrt(s*np.prod(s-self.dists))
        self.invariant = tuple(self.dists)
    
    def __str__(self):
        return f'{f'\nTriHash<{self.ref}>' if self.ref else f'TriHash'} {self.invariant}:\n{self.dists}\n{self.abc}\n'
    
    def __repr__(self):
        return f'{f'TriHash<{self.ref}>' if self.ref else f'TriHash'}:{self.invariant}'
        
    def __lt__(self, o):
        if isinstance(o,TriHash):
            return self.area < o.area
        
    def __hash__(self):
        return hash(self.invariant)
        
    def __eq__(self, o):
        return isinstance(o,TriHash) and o.invariant == self.invariant

def gauss_profile(X, Y, mean_x, mean_y, sig_x, sig_y, theta, A, B):
    X_off = X-mean_x
    Y_off = Y-mean_y
    sin,cos = np.sin(theta),np.cos(theta)
    X_rot = (X_off*cos - Y_off*sin)
    Y_rot = (Y_off*cos + X_off*sin)
    
    return A*np.exp(-np.square(X_rot/sig_x)/2-np.square(Y_rot/sig_y)/2)+B

def gauss_residual_fn(X,Y,Z):
    def fn(x):
        Z_fit = gauss_profile(X,Y,*x)
        return np.sum(np.square(Z-Z_fit))
    return fn

def find_stars(img, registration_channel=1, num_stars=250, patch_radius=10, verbose=False, 
               snr_min=10, max_peak_dist=2, min_fwhm=2, max_eccentricity=0.85, clip_cut=None,
               max_attempts=2000):
    '''
    Identify stars by looking for relatively isolated bright peaks and fitting 2d gaussian profiles.
    Handles multi-channel images by extracting one channel, working natively on a grayscale image/
    Success parameters are defined as keyword arguments to adjust the identified population,
    along with a total number of stars to identify and maximum number of candidates to consider.
    This can also create point spread functions from the identified stars.
    '''
    reg_ch = img[...,registration_channel] if registration_channel else img #None -> grayscale
    mean = np.mean(reg_ch)
    std = np.std(reg_ch)
    ch_max = np.max(reg_ch)

    reg_mask = np.zeros_like(reg_ch)
    shift = reg_ch-mean
    reg_mask[reg_ch > mean+std*3] = 1

    reg_blur = reg_mask
    for stage in range(2):
        reg_blur = gaussian_filter(reg_blur, sigma=3)
        
    reg_search = reg_blur.copy()

    attempts = 0
    candidate_stars = []

    i = 0
    while i < num_stars:
        idx = np.argmax(reg_search)
        x_max = idx%reg_search.shape[1]
        y_max = idx//reg_search.shape[1]
        if attempts >= max_attempts or reg_search[y_max,x_max] <= 0.0:
            print(f'Only {i} of {num_stars} requested stars from {attempts} attempts')
            break
        #print(x_max,y_max,reg_search[y_max,x_max])

        if x_max - patch_radius >= 0 and x_max + patch_radius < reg_search.shape[1] and y_max - patch_radius >= 0 and y_max + patch_radius < reg_search.shape[0]:
            reg_search[
                (y_max-patch_radius):(y_max+patch_radius+1),
                (x_max-patch_radius):(x_max+patch_radius+1)
            ] = 0 # zero out candidate region to skip in the future

            patch = reg_ch[(y_max-patch_radius):(y_max+patch_radius+1),(x_max-patch_radius):(x_max+patch_radius+1)]
            valid = np.logical_and(~np.isnan(patch), patch > 0)
            patch_valid = patch[valid]
            patch_max = np.max(patch_valid)
            if patch_max/np.min(patch_valid) < snr_min:
                continue # skip candidate as SNR too low to make the cut
            if clip_cut and patch_max > ch_max*clip_cut:
                continue # skip because there's clipping
                
            y = np.arange(y_max-patch_radius,y_max+patch_radius+1)
            x = np.arange(x_max-patch_radius,x_max+patch_radius+1)
            x,y = np.meshgrid(x,y)
            
            # fit the star profile
            fit = minimize(
                gauss_residual_fn(x[valid],y[valid],patch_valid), 
                (x_max,y_max,2,2,0,patch_max,np.mean(patch_valid)), 
                method='Powell'
            )

            attempts += 1

            mean_x, mean_y, sig_x, sig_y, theta, star_lvl, bkg_lvl = fit.x
            sig_x,sig_y = abs(sig_x),abs(sig_y)
            fwhm = np.sqrt(2*np.log(2))*(sig_x+sig_y)
            eccentricity = np.sqrt(1.0-np.square(min(sig_x,sig_y)/max(sig_x,sig_y)))
            dist = np.sqrt((x_max-mean_x)**2.0+(y_max-mean_y)**2.0)

            if verbose:
                print(f'Fit({fit.nfev}): {fit.message}')
                print(f'FWHM {fwhm} ECC {eccentricity} SNR {abs(star_lvl/bkg_lvl)} LVL {star_lvl} DIST {dist}')

            if (fit.success and 
                dist < max_peak_dist and 
                fwhm > min_fwhm and 
                eccentricity < max_eccentricity and 
                star_lvl > bkg_lvl and 
                abs(star_lvl/bkg_lvl) > snr_min and
                (clip_cut is None or star_lvl <= clip_cut*ch_max)
               ):
                
                candidate_stars.append([mean_x, mean_y, fwhm, star_lvl, bkg_lvl, eccentricity])
                i += 1
    
                if verbose:
                    plt.subplot(1,2,1)
                    plt.imshow(patch)
                    plt.colorbar()
                    plt.subplot(1,2,2)
                    plt.imshow(patch-gauss_profile(x,y,*fit.x))
                    plt.colorbar()
                    plt.show()
                    plt.close()
            else:
                if verbose:
                    print(f'Candidate {attempts} Excluded')
        else: #hit an edge
            reg_search[
                max(y_max-patch_radius,0):min(y_max+patch_radius+1,reg_search.shape[0]),
                max(x_max-patch_radius,0):min(x_max+patch_radius+1,reg_search.shape[1])
            ] = 0

    return np.asarray(candidate_stars)

def build_constellations(stars, k_nearest=7, **kwargs):
    pts = stars[:,:2]
    kdpts = KDTree(pts)
    dists,index = kdpts.query(pts, k=range(2,2+k_nearest))
    m,s = np.mean(dists.ravel()),np.std(dists.ravel())
    triangles = set()
    for x,(dist,idx) in enumerate(zip(dists,index)):
        mask = (dist > m/10) & (dist < m+3*s)
        p = pts[x]
        neighbors = kdpts.data[idx[mask]]
        tri = [[TriHash(p,a,b,**kwargs) for j,b in enumerate(neighbors) if j > i] for i,a in enumerate(neighbors)]
        for ts in [set(t) for t in tri if len(t)>0]:
            triangles |= ts
    return sorted(triangles)
        
def find_corresponding_constellations(tri_a,tri_b,fwhm=5,verbose=False):
    tri_a_lookup = np.asarray([t.dists for t in tri_a])
    tri_b_lookup = np.asarray([t.dists for t in tri_b])

    kdt_a = KDTree(tri_a_lookup)
    kdt_b = KDTree(tri_b_lookup)

    dists = []
    tri_a_match = []
    tri_b_match = []
    for i, matches in enumerate(kdt_a.query_ball_tree(kdt_b,fwhm)):
        if len(matches) == 1:
            j = matches[0]
            a = tri_a[i]
            b = tri_b[j]
            tri_a_match.append(a)
            tri_b_match.append(b)
    
    return tri_a_match, tri_b_match

tvec = lambda tri : tri.abc[0] - np.mean(tri.abc,axis=0)
norm = lambda x : x / np.sqrt(np.sum(np.square(x)))
dvec = lambda tri : norm(tvec(tri))
angle = lambda tri : np.atan2((dv:=dvec(tri))[1],dv[1])
mean_of_angles = lambda a : np.atan2(np.sum(np.sin(a)), np.sum(np.cos(a)))
stdev_of_angles = lambda a : np.sqrt(-2.0*np.log(np.mean(np.sqrt(np.square(np.sin(a))+np.square(np.cos(a))))))

def find_corresponding_points(tri_pair_a, tri_pair_b, angle_tol=5/180*np.pi):
    angle_diff = np.asarray([np.mod(angle(ta)-angle(tb), 2.0*np.pi) for ta,tb in zip(tri_pair_a,tri_pair_b)])
    mean_rot = mean_of_angles(angle_diff)
    mask = np.abs(angle_diff - mean_rot) <= angle_tol
    
    tri_pair_a = [x for x,y in zip(tri_pair_a,mask) if y]
    tri_pair_b = [x for x,y in zip(tri_pair_b,mask) if y]
    
    angle_diff = np.asarray([np.mod(angle(ta)-angle(tb), 2.0*np.pi) for ta,tb in zip(tri_pair_a,tri_pair_b)])
    mean_rot = mean_of_angles(angle_diff)
    
    corr_a = np.concatenate([t.abc for t in tri_pair_a])
    corr_b = np.concatenate([t.abc for t in tri_pair_b])
    
    rot_corr_b = transform_coords(corr_b, 0.0, 0.0, mean_rot)
    mean_tx = np.mean(corr_a - rot_corr_b, axis=0) # TODO filter again?
    
    corr_a, idx = np.unique(corr_a, axis=0, return_index=True)
    corr_b = corr_b[idx]
    corr_b, idx = np.unique(corr_b, axis=0, return_index=True)
    corr_a = corr_a[idx] 
    return corr_a, corr_b, mean_rot, mean_tx
