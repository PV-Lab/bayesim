import numpy as np
from fdint import fdk as fermi
import scipy.constants as const


def P3solver(P0,fs,r,Rl,dT,L,A,n,T0,Nv,A0,phi,Z):
    q = const.e
    kb = const.k
    kB = kb/q
    hbar = const.hbar
    T = T0+dT/2
    P = P0*((T/T0)**Z)
    eta = fs/(kB*T)
    
    S = kB * ((((r+2.5) * fermi(k=r+1.5,phi=eta)) / ((r+1.5) * fermi(k=r+1/2,phi=eta))) - eta)
    Rou = 1./Nv * 2 * const.pi**2 / (P*q**2) * (const.hbar**2/2.)**1.5 * (kb*T)**(-r-1.5) * 1/(2*r/3.+1) * 1/fermi(r+0.5,eta)
    a = Rou * L/A + Rl/n
    
    Il = (S*dT)/a
    y = Il**2 * Rl
    
    return y,S,Rou
    
def tefunnew(celldata,theta):
    # model function - using identical variable names to Danny's Matlab code
    # celldata should be a dict with keys 'xdata', 'ydata', 'n'
    # theta is just a list of length 4
    Temp0 = celldata['xdata']
    step = int(celldata['ydata'].shape[1]/2)
    Rl = celldata['ydata'][0,0:step]
    n = celldata['n']
    P0 = 10**theta[0]
    fs = theta[1]
    r = theta[2]
    Z = theta[3]
    
    Nv = 1
    phi = 0
    A0 = 0
    L = 1e-3 #length of device
    A = 0.25e-6 #area of device
    T0 = 300 #base temp
    
    P = np.zeros((len(Temp0),step))
    S = np.zeros((len(Temp0),step))
    Rou = np.zeros((len(Temp0),step))
    
    for i in range(len(Temp0)):
        for k in range(step):
            [P[i,k],S[i,k],Rou[i,k]] = P3solver(P0,fs,r,Rl[k],Temp0[i],L,A,n,T0,Nv,A0,phi,Z)
            
    y = P
    
    return y, S, Rou

def tefunnew_singlept(ec, params):
    """
    Function to calculate the TE device model at a single point to be used for the bayesim vesrion.
    
    Args:
        ec: experimental conditions, dictionary with keys T, Rl, n for temp., load resistance, # of "legs"
        params: dictionary with keys P0, fs, r, Z
    """
    # read in experimental conditions
    T = ec['T']
    Rl = ec['R']
    n = ec['n']
    
    # read in model parameters
    P0 = params['P0']
    fs = params['fs']
    r = params['r']
    Z = params['Z']
    
    # fixed values
    Nv = 1
    phi = 0
    A0 = 0
    L = 1e-3 #length of device
    A = 0.25e-6 #area of device
    T0 = 300 #base temp
    
    # return just the measured quantity to compare (first output of P3solver)
    return P3solver(P0, fs, r, Rl, T, L, A, n, T0, Nv, A0, phi, Z)[0]