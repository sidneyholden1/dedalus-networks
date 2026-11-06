"""Render and compare actual mixed-KdV Dedalus fields, without fitted pulses."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval


def sample(u,s):
    coefficients=dct(u[...,::-1],type=2,axis=-1)/u.shape[-1]
    coefficients[...,0]*=.5
    return chebval(2*np.asarray(s)-1,np.moveaxis(coefficients,-1,0))

