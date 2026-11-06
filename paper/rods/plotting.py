"""Independent physical-space comparison of saved warming solutions."""
import argparse,json
from pathlib import Path
import numpy as np
from numpy.polynomial import chebyshev as cheb


def evaluate(report, snapshot, z):
    L=np.array(report['lengths']);K=np.array(report['K']);a=np.array(report['a'])
    edges=np.array(report['edges']);W=np.array(report['steady_vertices'])
    k=np.sqrt(a/K)*L;s=(z+1)/2
    steady=(W[edges[:,0],None]*np.sinh(k[:,None]*(1-s))+
            W[edges[:,1],None]*np.sinh(k[:,None]*s))/np.sinh(k[:,None])
    return np.array([cheb.chebval(z,c) for c in snapshot['coefficients']])+(-np.expm1(-4*snapshot['time']**2))*steady

