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
import concurrent.futures as futures

def simple_stack(img_set, selection=None, verbose=False, **kwargs):
    '''
    Stacks an iterable caled img_set which is assumed to have the properties 
    img_shape and num_img defined.
    
    This is done with numpy's mean function, which seems to be relatively memory
    efficient as long as the iterable loads lazily. oscdeeppy.image sets satisfy 
    these requirements.
    '''
    if selection is None:
        return np.mean(img_set, axis=0, **kwargs)
    else:
        return np.mean((img for img,use in zip(img_set,selection) if use), axis=0, **kwargs)
        
        
class _winsor_chunk:
    '''
    Helper class for winsor_stack to wrap a unit of work processing one chun for multiprocessing.
    '''
    
    def __init__(self, img_set, winsor_low, winsor_high):
        self.img_set = img_set
        self.winsor_low = winsor_low
        self.winsor_high = winsor_high
        
    def __call__(self, x_i, y_i, x_span, y_span):
        img_slice = np.asarray([img[y_i:y_i+y_span,x_i:x_i+x_span] for img in self.img_set])
        quantiles = np.quantile(img_slice,[self.winsor_low,1.0-self.winsor_high],axis=0)
        return (x_i, y_i, x_span, y_span),quantiles

def winsor_stack(img_set, winsor_low=0.03, winsor_high=0.03, selection=None, span = 1000, nproc=4, verbose=False):
    '''
    Similar to simple_stack, but Winsorizes the pixel data to clip outliers 
    to the quantile specified by winsor_high and windsor_low.
    
    Processes images in chunks, since the quantile calculation is RAM intensive.
    Chunk size is controlled by span, and multiprocessing can be tuned with nproc workers.
    '''
    if verbose:
        print(f'Computing quantiles [{winsor_low:0.03f}, {1.0-winsor_high:0.03f}] for winsorizing')
    low,high = np.empty(img_set.img_shape,dtype=img_set.dtype),np.empty(img_set.img_shape,dtype=img_set.dtype)
    _chunk = _winsor_chunk(img_set,winsor_low,winsor_high)
    with futures.ProcessPoolExecutor(nproc) as pool:
        f_queue = set()
        for x_i in range(0,img_set.img_shape[1],span):
            for y_i in range(0,img_set.img_shape[0],span):
                x_span = span if x_i + span <= img_set.img_shape[1] else img_set.img_shape[1]-x_i
                y_span = span if y_i + span <= img_set.img_shape[0] else img_set.img_shape[0]-y_i
                
                if verbose:
                    print(f'Submitting chunk ({x_i}, {y_i})')
                
                f = pool.submit(_chunk, x_i, y_i, x_span, y_span)
                f_queue.add(f)
                
                all_queued = x_i+x_span+1 >= img_set.img_shape[1] and y_i+y_span+1 >= img_set.img_shape[0]
                buffer_full = len(f_queue) >= nproc + nproc//2
                if buffer_full or all_queued:
                    if all_queued:
                        done = futures.as_completed(f_queue)
                    else:
                        done,f_queue = futures.wait(f_queue,return_when=futures.FIRST_COMPLETED)
                    while True:
                        for f_done in done:
                            (res_x_i, res_y_i, res_x_span, res_y_span),quantiles = f_done.result()
                            low[res_y_i:res_y_i+res_y_span,res_x_i:res_x_i+res_x_span] = quantiles[0]
                            high[res_y_i:res_y_i+res_y_span,res_x_i:res_x_i+res_x_span] = quantiles[1]
                            if verbose:
                                print(f'Finished chunk ({res_x_i}, {res_y_i})')
                        if all_queued:
                            break
                        else:
                            done,f_queue = futures.wait(f_queue,timeout=0.5,return_when=futures.FIRST_COMPLETED)
                            if len(done) == 0:
                                break 
    if verbose:
        print(f'Stacking images')
    result = np.zeros(img_set.img_shape, img_set.dtype)
    if selection is None:
        selection = [True]*len(img_set)
    for i, (img,use) in enumerate(zip(img_set,selection)):
        if not use:
            if verbose:
                print(f'Skipping image {i}')
            continue
        if verbose:
            print(f'Processing image {i}')
        result += np.clip(img,low,high)
    return result/np.count_nonzero(selection)
