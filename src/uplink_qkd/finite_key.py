"""This file holds the optimisation functions for the finite key BBM92 protocol.
Described in the C. Lim paper: https://doi.org/10.1103/PhysRevLett.126.100501"""
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def h(x):
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

def objective(x, delta, m, eps_qkd, t, f):
    beta, nu, xi = x
    k = np.floor(beta * m)
    n = m - k
    gamma = 1/(1 + m*(delta + xi)) + 1/(1 + m - m*(delta + xi))
    eps_pe =  np.sqrt(np.exp(-2*m*k*xi**2 / (n+1)) + np.exp(-2 * gamma *((n*(nu-xi))**2-1)) )
    r = f * n * h(delta)  # this factor n is not there in the paper
    l = 2*np.log2(2*(eps_qkd-2**(-t)-2*eps_pe)) + n*(1-h(delta+nu)) - r - t
    return l / m


def brute_search_parallel_equality(m, delta, eps_qkd, t, f, granularity=200):
    """maximize alpha respecting constraint, walk closer and closer to the constraint, 
    doing a brute search of the other three params until the constraint is satisfied"""
    beta_range = np.linspace(0,0.5,granularity)
    nu_range = np.linspace(0,0.5-delta,granularity)
    xi_range = np.linspace(0,0.5-delta,granularity)

    beta_array = np.repeat(beta_range, len(nu_range)*len(xi_range))
    nu_array = np.tile(np.repeat(nu_range, len(xi_range)), len(beta_range))
    xi_array = np.tile(xi_range, len(beta_range)*len(nu_range))
    
    alphas = objective([beta_array,nu_array,xi_array],delta, m, eps_qkd, t, f)
    mask = np.where(np.logical_and(np.logical_and(np.logical_and(alphas>0, alphas<1-beta_array), ~np.isnan(alphas)),nu_array>xi_array))[0]
    try:
        index= np.argmax(alphas[mask])
        return alphas[mask][index], beta_array[mask][index], nu_array[mask][index], xi_array[mask][index]
    except:
        return 0,0,0,0
    
# The actual optimisation
def neg_objective(x, delta, m, eps_qkd, t, f):
    return -objective(x, delta, m, eps_qkd, t, f)


def smart_optimise(m, delta, eps_qkd, t, f, granularity=200):
    init_vals = brute_search_parallel_equality(m, delta, eps_qkd, t, f, granularity)[1:]
    constraints = [
    {'type': 'ineq', 'fun': lambda x: 0.5-abs(x[0])},  # beta <= 0.5 
    {'type': 'ineq', 'fun': lambda x: 0.5-delta-x[1]},  #  nu <= 0.5-delta
    {'type': 'ineq', 'fun': lambda x: 0.5-delta-x[2]},  #  xi <= 0.5-delta
    {'type': 'ineq', 'fun': lambda x: x[1]-x[2]},  # nu >= xi
    ]
    result = minimize(neg_objective,x0=init_vals,constraints=constraints,args=(delta, m, eps_qkd, t, f),method="Nelder-Mead")
    brute_result = objective(init_vals, delta, m, eps_qkd, t, f) 
    if np.isnan(brute_result):  brute_result=0
    #Nelder Mead works after a brute coarse search, not on its own
    if -result.fun>=brute_result and np.all(np.array([0.5-abs(result.x[0]),0.5-delta-result.x[1],0.5-delta-result.x[2],result.x[1]-result.x[2]] > np.array([0, 0, 0, 0]))):
        if np.isnan(result.fun):  return 0
        return  -result.fun
    return brute_result
