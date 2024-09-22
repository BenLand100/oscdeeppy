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

import os
import concurrent.futures as futures

import matplotlib.pyplot as plt
import numpy as np

import rawpy
import fitsio

from tqdm import tqdm

class FITSSet:
    '''
    A collection of fits image files on disk (read-only). This would work for any
    raw type supported by fitsio, but RGGB bayer pattern is assumed in some of the 
    code. 
    
    Images are loaded on the fly, so this is safe across multiprocessing, etc.
    '''
    
    def __init__(self, files):
        self.files = files
        self.num_img = len(files)
        test = fitsio.read(files[0])
        self.img_shape = test.shape
        self.dtype = test.dtype
    
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, i):
        assert type(i) == int, 'Must assign to integer values only'
        return fitsio.read(self.files[i]) 
        
    def lazy_read(self, i):
        return IndirectImg(self,i)

class RawSet:
    '''
    A collection of raw image files on disk (read-only). This would work for any
    raw type supported by rawpy, but RGGB bayer pattern is assumed in some of the 
    code. 
    
    Images are loaded on the fly, so this  is safe across multiprocessing, etc.
    '''
    
    def __init__(self, files):
        self.files = files
        self.num_img = len(files)
        with rawpy.imread(files[0]) as test:
            self.img_shape = test.raw_image.shape
            self.dtype = test.raw_image.dtype
    
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, i):
        assert type(i) == int, 'Must assign to integer values only'
        with rawpy.imread(self.files[i]) as fraw:
            return fraw.raw_image.copy()

    def lazy_read(self, i):
        return IndirectImg(self,i)
        
class DiskSet:
    '''
    A "custom format" for storing intermediate image data on disk as a single file.
    This is achieved with a memory mapped numpy ndarray.
    
    The memmap is not pickled, and properly reconstructed on unpickling, so this
    works nicely with multiprocessing.
    '''
    
    def __init__(self, fname, recreate=False, num_img=None, img_shape=None, dtype=np.float32):
        self.fname = fname
        assert num_img is not None, 'DiskSet requires num_img'
        assert img_shape is not None, 'DiskSet requires img_shape'
        self.img_shape = img_shape
        self.num_img = num_img
        self.set_shape = (num_img,)+img_shape
        self.dtype = dtype
        if os.path.exists(fname) and not recreate:
            self._mmap = np.memmap(fname, dtype=dtype, mode='r+', shape=self.set_shape)
        else:
            self._mmap = np.memmap(fname, dtype=dtype, mode='w+', shape=self.set_shape)
            self._mmap.flush()

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['_mmap']
        return d
        
    def __setstate__(self, d):
        self.__dict__.update(d)
        self._mmap = np.memmap(self.fname, dtype=self.dtype, mode='r+', shape=self.set_shape)
        
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, i):
        return self._mmap[i]
        
    def lazy_read(self, i):
        return IndirectImg(self,i)
        
    def __setitem__(self, i, val):
        self._mmap[i] = val
        self._mmap.flush()
            
    def lazy_write(self, i):
        return IndirectImg(self,i)
        
class IndirectImg:
    '''
    Encapsulates a reference to a specific image in a set in a way that is
    safe for multiprocessing.
    '''
    
    def __init__(self, img_set, idx):
        self.img_set = img_set
        self.idx = idx

    def load(self):
        return self.img_set[self.idx]

    def save(self, img):
        self.img_set[self.idx] = img
        
def _iproc_worker(func, i, in_img, out_img, iter_args, args, kwargs):
    '''Worker method for ImageProcessor multiprocessing representing one unit of work.'''
    if type(in_img) is tuple: # reduce mode
        img_a,img_b = in_img
        if img_a is None:
            return i,img_b.load()
        else: 
            img_a = img_a.load()
        if img_b is None:
            return i,img_a.load()
        else:
            img_b = img_b.load()
        result = func(img_a,img_b,*iter_args,*args,**kwargs)
    else: # map mode
        img = in_img.load()
        result = func(img,*iter_args,*args,**kwargs)
    if out_img:
        out_img.save(result)
        return i,None
    else:
        return i,result
    
class ImageProcessor:
    '''
    Abstracts out multiprocessing and memory efficient patterns for processing 
    lists of images. Map functionality is implemented allowing functions to be 
    applied to all elements in an image list, with optional additional parameters. 
    In this mode the output can be another image set to avoid returning large 
    results. Reduce functionality is implemented as well, but is not heavily used.
    
    By allowing input from and output to abstracted image sets, only necesasry
    data can be loaded into memory at any given point, and large amounts of data
    need not be passed between processes for mulitprocessing. 
    '''
    

    def __init__(self, nproc=4, buffer=None):
        self.nproc = nproc
        self.buffer = nproc//2
        self.pool = futures.ProcessPoolExecutor(max_workers=nproc)

    def map(self, func, input_set, *iter_args, args=[], selection=None, output_set=None, reduce=None, verbose=False, **kwargs):
        if output_set is None:
            if reduce is None:
                results = [None]*len(input_set)
            else:
                results = None
        if verbose:
            total_jobs = len(input_set) if selection is None else np.count_nonzero(selection)
            pbar = tqdm(total=total_jobs)
        fs = set()
        args_it = zip(*iter_args) if iter_args else None
        j = 0
        for i in range(len(input_set)):
            if selection is None or selection[i]:
                in_img = input_set.lazy_read(i)
                out_img = output_set.lazy_write(j) if output_set is not None else None
                if verbose:
                    pass
                    #print(f'Starting {i}->{j}')
                f = self.pool.submit(_iproc_worker, func, j, in_img, out_img, next(args_it) if args_it else [], args, kwargs)
                fs.add(f)
                j = j + 1
            else:
                if verbose:
                    pass
                    #print(f'Skipping {i}')
                next(args_it)
            all_queued = i+1 == len(input_set)
            buffer_full = len(fs) >= self.nproc+self.buffer
            if buffer_full or all_queued:
                if all_queued:
                    done = futures.as_completed(fs)
                else:
                    done,fs = futures.wait(fs,return_when=futures.FIRST_COMPLETED)
                for f_done in done:
                    j_done, res_done = f_done.result()
                    if verbose:
                        pbar.update(1)
                        #print(f'Finished {j_done}')
                    if reduce is None:
                        if output_set is None:
                            results[j_done] = res_done
                    else:
                        if results is None:
                            results = res_done
                        else:
                            results = reduce(results,res_done)
        if verbose:
            pbar.close()
        if output_set is None:
            return results

    def reduce(self, func, input_set, args=[], dtype=np.float64, selection=None, verbose=False, **kwargs):
        fs = set()
        waiting_result = None
        i = 0
        k = 0
        while i < len(input_set):
            in_imgs = [None,None]
            for j in range(2):
                if waiting_result is not None:
                    in_imgs[j] = IndirectImg([waiting_result],0)
                    waiting_result = None
                elif i < len(input_set):
                    if selection is None or selection[i]:
                        if verbose:
                            print(f'Starting {i}')
                        in_imgs[j] = input_set.lazy_read(i)
                        i = i+1
                    else:
                        if in_imgs[0] is not None:
                            # just in case something was set here
                            waiting_result = in_imgs[0].load() 
                        in_imgs = None
                        if verbose:
                            print(f'Skipping {i}')
                        i = i+1
                        break
                else:
                    break

            if in_imgs is not None:
                f = self.pool.submit(_iproc_worker, func, i, tuple(in_imgs), None, [], args, kwargs)
                fs.add(f)
            
            all_queued = i+1 >= len(input_set)
            while len(fs) >= self.nproc+self.buffer or (all_queued and len(fs) > 0):
                if all_queued:
                    done = futures.as_completed(fs)
                    fs = set()
                else:
                    done,fs = futures.wait(fs,return_when=futures.FIRST_COMPLETED)
                    
                for f_done in done:
                    i_done, res_done = f_done.result()
                    res_done = np.asarray(res_done, dtype=dtype)
                    if verbose:
                        print(f'Finished {i_done}')
                    if waiting_result is None:
                        waiting_result = res_done
                    else:
                        in_imgs = ( IndirectImg([waiting_result],0), IndirectImg([res_done],0) )
                        waiting_result = None
                        ki = len(input_set) + k
                        k = k+1
                        if verbose:
                            print(f'Starting {ki}')
                        f = self.pool.submit(_iproc_worker, func, ki, in_imgs, None, [], args, kwargs)
                        fs.add(f)
        return waiting_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown(wait=False)
    
    def __del__(self):
        self.pool.shutdown(wait=False)
