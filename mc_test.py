import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from mc_util import *

def main():
    np.random.seed(1)

    # as an example...
    U = np.array([[1,0,0,1],[2,0,1,0]]).T
    V = np.array([[1,1,0,0],[0,0,1,1]]).T
    print('two low rank matrices are:')
    print('U = ')
    print(U)
    print('V = ')
    print(V)
    M = U @ V.T
    print('these create the low rank matrix M:')
    print(M)
    Omega_mask = np.array([[1,      np.nan, 1,      np.nan],
                           [np.nan, np.nan, np.nan, 1],
                           [np.nan, 1,      1,      np.nan],
                           [1,      1,      np.nan, np.nan]])
    print('the values we observe follow the mask:')
    print(Omega_mask * M)
    Omega_i, Omega_j = matIndicesFromMask(Omega_mask)

    Vop = matricize_right(V, Omega_mask)
    M_Omega_recovered = Vop @ vec(U)
    M_Omega = masked(M, Omega_mask)

    print('Vop.shape = {}'.format(Vop.shape))
    print('vec(U).shape = {}'.format(vec(U).shape))
    print('M_Omega_recovered.shape = {}'.format(M_Omega_recovered.shape))
    print('rmse = {}'.format(rmse(M_Omega_recovered, M_Omega)))


    Uop = matricize_left(U,Omega_mask)
    M_Omega_recovered = Uop @ vec(V)
    M_Omega = masked(M, Omega_mask)
    
    print('Uop.shape = {}'.format(Uop.shape))
    print('vec(V).shape = {}'.format(vec(V).shape))
    print('M_Omega_recovered.shape = {}'.format(M_Omega_recovered.shape))
    print('rmse = {}'.format(rmse(M_Omega_recovered, M_Omega)))



    print('altMinSense with LS')
    U_ls, V_ls = altMinSense(M_Omega=M_Omega,
                             Omega_mask=Omega_mask,
                             r=2, method='lsq')
    print(U_ls)
    print(V_ls)
    print(rmse(U_ls @ V_ls.T, M))

    print('altMinSense with CVX')
    U_cvx, V_cvx = altMinSense(M_Omega=M_Omega,
                               Omega_mask=Omega_mask,
                               r=2, method='cvx')
    print(U_cvx)
    print(V_cvx)
    print(rmse(U_cvx @ V_cvx.T, M))

    return

if __name__ == "__main__":
    main()
