from cvxpy import Minimize, Problem, Variable, SCS
from cvxpy import mul_elemwise
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec

from scipy.optimize import least_squares
from scipy.linalg import norm as spnorm

import numpy as np

from mc_util import *


def mcFrobSolveRightFactor_cvx(U, M_Omega, mask, **kwargs):
    """
    A solver for the right factor, V, in the problem 
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    n = mask.shape[1]
    r = U.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)
    
    # Problem
    V_T = Variable(r, n)
    obj = Minimize(cvxnorm(cvxvec((U @ V_T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    V = V_T.value.T
    if returnObjectiveValue:
        return (V, prob.value)
    else:
        return V


def mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs):
    """
    mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs)
    A solver for the left factor, U, in the problem
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    m = mask.shape[0]
    if V.shape[0] < V.shape[1]:
        # make sure V_T is "short and fat"
        V = V.T
    r = V.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)

    # Problem
    U = Variable(m, r)
    obj = Minimize(cvxnorm(cvxvec((U @ V.T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    if returnObjectiveValue:
        return (U.value, prob.value)
    else:
        return U.value


def mcFrobSolveLeftFactor_ls(V, M_Omega, mask, **kwargs):
    r = V.shape[1]
    Vop = matricize_right(V, mask)
    
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', 1)
    x0 = kwargs.get('x0', np.random.rand(Vop.shape[1]))
    
    def frob_error(x, Vop, M_Omega):
        return spnorm((Vop @ x) - M_Omega)

    ls = least_squares(frob_error, x0=x0, kwargs={'Vop' : Vop, 'M_Omega': M_Omega}, verbose=1)
    if returnObjectiveValue:
        return (unvec(ls.x, (-1, r)), ls.cost)
    else:
        return unvec(ls.x, (-1, r))


def mcFrobSolveRightFactor_ls(U, M_Omega, mask, **kwargs):
    """
    mcFrobSolveRightFactor_ls(U, M_Omega, mask, **kwargs)
    solves for the right factor, V, using least squares.
    """
    r = U.shape[1]
    Uop = matricize_left(U, mask)

    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', 1)
    x0 = kwargs.get('x0', np.random.rand(Uop.shape[1]))
    
    def frob_error(x, Uop, M_Omega):
        return spnorm((Uop @ x) - M_Omega)

    ls = least_squares(frob_error, x0=x0,
                       kwargs={'Uop' : Uop, 'M_Omega': M_Omega})
    if returnObjectiveValue:
        return (unvec(ls.x, (-1, r)), ls.cost)
    else:
        return unvec(ls.x, (-1, r))


def altMinSense(M_Omega, Omega_mask, r, **kwargs):
    """
    altMinSense(M_Omega, Omega_mask, r, **kwargs)
    The alternating minimization algorithm for a matrix completion
    version of the matrix sensing problem
    
    Input
    max_iters : the maximum allowable number of iterations of the algorithm
    optCond : the optimality conditions that is measured 
              (default: absolute difference)
    optTol : the optimality tolerance used to determine stopping conditions
    solveLeft : a function to solve for the left matrix, Uj, on iteration j
                (default: mcFrobSolveLeftFactor_cvxpy)
    solveRight : a function to solve for the right matrix, Vj, on iteration j
                (default: mcFrobSolveRightFactor_cvxpy)
    solver : which solver to use (for cvxpy only) (default: SCS)
    verbose : 0 (none), 1 (light, default) or 2 (full) level of verbosity

    Ouptut
    U : the left m-by-r factor
    V : the right n-by-r factor
    """
    max_iters = kwargs.get('max_iters', 50)
    method = kwargs.get('method', 'cvx')
    optCond = kwargs.get('optCond', lambda x, y: np.abs(x - y))
    optTol = kwargs.get('optTol', 1e-4)
    solveLeft = kwargs.get('leftSolve', None)
    solveRight = kwargs.get('rightSolve', None)
    opts = kwargs.get('methodOptions', None)
    verbose = kwargs.get('verbose', 1)

    if method == 'lsq':
        solveLeft = mcFrobSolveLeftFactor_ls
        solveRight = mcFrobSolveRightFactor_ls
        if opts is None:
            opts = {'verbose' : verbose}
    elif method == 'cvx':
        solveLeft = mcFrobSolveLeftFactor_cvx
        solveRight = mcFrobSolveRightFactor_cvx
        if opts is None:
            opts = {'solver': SCS, 'verbose': verbose}
        elif opts.get('solver') is None:
            opts['solver'] = SCS

    if not verbose:
        verbose = False
        verbose_solve = False
    elif (verbose is True) or (verbose == 1):
        verbose = True
        verbose_solve = False
    elif (verbose == 2):
        verbose = True
        verbose_solve = True

    m, n = Omega_mask.shape
    # # Create initial guess from unbiased estimator # #
    # Set initial entries of estimator
    unbiased = np.zeros(Omega_mask.shape)
    unbiased[matIndicesFromMask(Omega_mask)] = M_Omega
    # scale entries of estimator by an estimate on the sampling
    # probability p so that this estimator is unbiased
    unbiased /= (M_Omega.size / Omega_mask.size)
    # compute svd of the unbiased estimator
    unbiased_left, unbiased_sing, unbiased_right = np.linalg.svd(unbiased)
    U = unbiased_left[:, :r]
    objPrevious = np.inf
    for T in range(max_iters):
        V = solveRight(U, M_Omega, Omega_mask, **opts)
        U, objValue = solveLeft(V, M_Omega, Omega_mask, **opts,
                                returnObjectiveValue=True)
        
        if optCond(objValue, objPrevious) < optTol:
            print()
            print('Optimality conditions satisfied.')
            print('Objective value = {:5.3g}'.format(objValue))
            break
        else:
            if verbose:
                print('Iteration {}: Objective = {}'.format(T, objValue), end='\r')
            objPrevious = objValue
    return U, V

