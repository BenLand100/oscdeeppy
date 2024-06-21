# OSCDeepPy

This is a collection of python utilities I put together to process images of the
night sky taken with standard One-Shot-Color (OSC) digital cameras containing a 
[Bayer Filter](https://en.wikipedia.org/wiki/Bayer_filter). The processing is
intended to combine multiple long exposure images for the purpose of extracting
an image of faint [Deep-sky Objects](https://en.wikipedia.org/wiki/Deep-sky_object)
from light polluted raw data. This is all done internally in 32bit floating point.

This process combines:
* "Light" images, taken as long exposures of the intended target.
* "Dark" images, nominally under the same settings and conditions as the "Light" 
  images, but possibly just intermediate exposure lengths to be rescaled. 
  For these, the lens cap (or equiv) is left on, so that no real light reaches
  the camera. Corrects for integrated electronics noise.
* "Bias" images, as close to zero-time exposures as possible, subtracted off
  all images to correct for offsets.
* "Flat" images, taken as proper exposures of a perfectly uniform flat field. 
  These are used to correct for color and sensitivity of the optical system, due
  to sensor variation, vignetting, dust, etc.
  
To obtain: a true-to-life image of deep space objects as if they were brighter 
or we had better vision to see them with.
  
Contains Python(+Numpy+Scipy) implementations of:
* Debayering - but be warned, I guessed at how this works and need to research more
* Simple and Winsorized stacking - RAM efficient & multiprocessing
* Star and constellation (triangles of stars) identification 
* Efficient KD-tree search (using Scipy) of similar triangles in different images
* Alignment optimization, including pretty aggressive outlier culling for better fits
* Image transformation, including 2X "drizzle", with bilinear interpolation - RAM efficient & multiprocessing
* Automated polynomial background estimation and elimination
* Automated post-processing with sane defaults for Generalized Hyperbolic Stretching
* Export of 16-bit PNGs via OpenCV - this was poorly supported in Python APIs for other pacakages

Notable TODO:
* There is no UI; full auto w/ tunable parameters. Might need to chage that.
* Gradient removal with Radial Basis Function, which I know better as Kernel Density Functions, 
  as a final step is still necessary. I do this in Siril with the 16bit PNG.
* Significant color tweaking. So far I've found this unnecessary, as these OSC
  images are somewhat balanced, provided light pollution or light leak aren't
  too significant in gradient removal.
  
See example.html for usage, and stay tuned as this evolves into a proper Python
package. For now, a stab at a requirements.txt file exists.

The code is released under the GPLv3; however, the copyright to any images in 
this repo is held by myself, Benjamin Land, and no permissions are granted to 
use or redistribute.
