from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def OpenFits(file, BoW=0):
    f = fits.open(file)
    d = f[0].data                    # 2-D array with image
    vminVal = 0.95*np.median(d)      # may need to adjust
    vmaxVal = 1.1*np.median(d)
    cmapVal = 'gray'                 # white stars on black background
    if BoW == 0:
        cmapVal = 'Greys'            # black stars on white background
    # plt.imshow(d, vmin=vminVal, vmax=vmaxVal, cmap=cmapVal, origin='upper')
    # plt.show()
    return d
