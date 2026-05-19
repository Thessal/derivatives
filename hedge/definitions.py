import numpy as np
import torch
from torch.distributions.normal import Normal
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

sim_step = 10
bs = 1000
S0 = 100
sigma = 0.01
# r = 0 #0.001 # risk free rate 
r = 0.5 * sigma * sigma
q = 0 # dividend payout rate
E = 100
T = 1
dt = T / sim_step

# log price of underlying
X = Normal((r - q - 0.5 * sigma * sigma) * dt, sigma * np.sqrt(dt))
# normal distribution CDF
N = lambda z : (torch.erf(z * np.sqrt(0.5)) + 1 ) * 0.5

def bsm(x, ttm, E):
    S = torch.exp(x)
    K = np.log(E)
    d1 = (x - K + r - q + sigma * sigma * 0.5 * ttm) / sigma / torch.sqrt(ttm)
    d2 = (x - K + r - q - sigma * sigma * 0.5 * ttm) / sigma / torch.sqrt(ttm)
    # BSM call price 
    C = S * torch.exp(-q * ttm) * N(d1) - E * torch.exp(-r * ttm) * N(d2) 
    C[ttm==0] = torch.maximum(S[ttm==0] - E, torch.tensor(0.0, device=S.device))
    return C

def bsm_delta(x, ttm, E):
    S = torch.exp(x)
    K = np.log(E)
    d1 = (x - K + r - q + sigma * sigma * 0.5 * ttm) / sigma / torch.sqrt(ttm)
    delta = torch.exp(-q * ttm) * N(d1)
    
    # Handle expiration (ttm == 0)
    is_expired = (ttm == 0)
    if is_expired.any():
        delta[is_expired] = torch.where(S[is_expired] > E, torch.tensor(1.0, device=S.device), torch.tensor(0.0, device=S.device))
        
    return delta

def sample():
    x = X.sample(sample_shape=[sim_step,bs])
    x[0,:] = 0
    x = x.cumsum(0) + np.log(S0)
    ttm = T - torch.vstack([torch.linspace(0,T,sim_step)]*bs).T
    C = bsm(x, ttm, E)
    dC = C.diff(axis=0)
    S = torch.exp(x)
    dS = S.diff(axis=0)

    return x[:-1], ttm[:-1], dC, dS, S[:-1]#S[1:]