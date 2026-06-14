"""Finite-key optimisation routines for the BBM92 protocol.

This module contains the optimisation functions used to evaluate the finite-key
secret key length ratio for the BBM92 protocol. The implementation follows the
finite-key expression described in C. C. W. Lim et al.,
Phys. Rev. Lett. 126, 100501 (2021):

https://doi.org/10.1103/PhysRevLett.126.100501

The optimisation proceeds by first performing a coarse brute-force search over
the optimisation parameters and then refining the result using a local
minimisation routine.

"""
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def h(x):
    """
    Calculate the binary entropy function.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input probability or array of probabilities.

    Returns
    -------
    entropy : float or numpy.ndarray
        Binary entropy evaluated at ``x``.

    """
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

def objective(x, delta, m, eps_qkd, t, f):
    """
    Calculate the finite-key secret key length ratio objective.

    Parameters
    ----------
    x : list or numpy.ndarray
        Optimisation parameters ``[beta, nu, xi]``.
    delta : float
        Quantum bit error rate used in the finite-key expression.
    m : float
        Total block size.
    eps_qkd : float
        Security parameter for the QKD protocol.
    t : float
        Error-verification parameter appearing in the finite-key expression.
    f : float
        Error-correction efficiency factor.

    Returns
    -------
    key_length_ratio : float or numpy.ndarray
        Secret key length divided by the total block size ``m``.

    """
    beta, nu, xi = x
    k = np.floor(beta * m)
    n = m - k
    gamma = 1/(1 + m*(delta + xi)) + 1/(1 + m - m*(delta + xi))
    eps_pe =  np.sqrt(np.exp(-2*m*k*xi**2 / (n+1)) + np.exp(-2 * gamma *((n*(nu-xi))**2-1)) )
    r = f * n * h(delta)  # this factor n is not there in the paper
    l = 2*np.log2(2*(eps_qkd-2**(-t)-2*eps_pe)) + n*(1-h(delta+nu)) - r - t
    return l / m


def brute_search_parallel_equality(m, delta, eps_qkd, t, f, granularity=200):
    """
    Perform a coarse brute-force search over the optimisation parameters.

    Parameters
    ----------
    m : float
        Total block size.
    delta : float
        Quantum bit error rate used in the finite-key expression.
    eps_qkd : float
        Security parameter for the QKD protocol.
    t : float
        Error-verification parameter appearing in the finite-key expression.
    f : float
        Error-correction efficiency factor.
    granularity : int, optional
        Number of grid points used for each optimisation-parameter range.

    Returns
    -------
    alpha : float
        Largest positive secret key length ratio found by the brute-force
        search.
    beta : float
        Value of ``beta`` associated with ``alpha``.
    nu : float
        Value of ``nu`` associated with ``alpha``.
    xi : float
        Value of ``xi`` associated with ``alpha``.

    """
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
    """
    Calculate the negative finite-key objective for minimisation.

    Parameters
    ----------
    x : list or numpy.ndarray
        Optimisation parameters ``[beta, nu, xi]``.
    delta : float
        Quantum bit error rate used in the finite-key expression.
    m : float
        Total block size.
    eps_qkd : float
        Security parameter for the QKD protocol.
    t : float
        Error-verification parameter appearing in the finite-key expression.
    f : float
        Error-correction efficiency factor.

    Returns
    -------
    negative_key_length_ratio : float or numpy.ndarray
        Negative of the secret key length ratio returned by ``objective``.

    """
    return -objective(x, delta, m, eps_qkd, t, f)


def smart_optimise(m, delta, eps_qkd, t, f, granularity=200):
    """
    Optimise the finite-key secret key length ratio.

    Parameters
    ----------
    m : float
        Total block size.
    delta : float
        Quantum bit error rate used in the finite-key expression.
    eps_qkd : float
        Security parameter for the QKD protocol.
    t : float
        Error-verification parameter appearing in the finite-key expression.
    f : float
        Error-correction efficiency factor.
    granularity : int, optional
        Number of grid points used for the initial brute-force search.

    Returns
    -------
    optimal_key_length_ratio : float
        Optimised secret key length divided by the total block size ``m``.

    """
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
