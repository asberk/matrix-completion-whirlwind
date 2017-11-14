from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, lil_matrix, coo_matrix
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def subsampledMatrix(M, dist=None, **kwargs):
    """
    subsampledMatrix(M, p=.1, dist[=None][, observed]) returns a 
        subsampled version of the input matrix M.
    
    Input
    dist : the distribution according to which the entries are sampled.
           options: unif, None
    kwargs (optional)
    p : input parameter when dist='unif' specifies probability that 
        entries are observed.
    observed : (when dist=None) the entries of the array that were observed. 
               Note: pass (i,j) values as 
                     [[i1, i2, i3, ...], [j1, j2, j3, ...]]
    Output: the subsampled matrix M
    """
    if dist == 'unif':
        p = kwargs.get('p', .1)
        returnMaskAsIndices = kwargs.get('returnMaskAsIndices', False)
        mask = np.random.rand(*M.shape) <= p
        if returnMaskAsIndices:
            return (M * mask, [x.tolist() for x in np.where(mask)])
        else:
            return M * mask
    elif dist is None:
        observedEntries = kwargs.get('observed', None)
        ret = np.zeros(M.shape)
        ret[observedEntries] = M[observedEntries]
        return ret
    else:
        return


def rmse(A, B):
    """
    rmse(A, B) is a function to compute the relative mean-squared
    error between two matrices A and B, given by
    1/(m*n) * sum_ij (A_ij - B_ij)^2
    where A and B are m-by-n numpy arrays.
    """
    return np.sqrt(mean_squared_error(A,B))

def matrixCompletionSetup(r, m, n=None, p=None):
    """
    matrixCompletionSetup(m, n, r, p) computes everything necessary to
                                      be set-up for a matrix
                                      completion problem using the
                                      structural assumption that M has
                                      low rank and each entry is
                                      observed independently with
                                      probability p.
    Input
    m : the number of rows of the output matrix M
    n : the number of columns of the output matrix M
    r : the rank of the output matrix M. 
        Note that this value can be a vector of values. 
    p : the observation probability for entries of M. 
        Note that this value can be a vector of values.
    k : the number of iterates to compute (default 1)

    Output
             U : the left m-by-r matrix
             V : the right n-by-r matrix
         Omega : the list of indices of M that were observed
    Omega_mask : the mask corresponding to observed entries Omega of
                 the matrix M
    """
    print('There has been a re-write of this function. Please ' +
          'check documentation or source for more information.' +
          ' (cf. sparseMatComSetup for a sparse version of this ' + 
          'function.)')
    if n is None:
        n = m
    if p is None:
        p = .5

    U = np.random.randint(0, 5, size=(m,r))
    V = np.random.randint(0, 5, size=(n,r))
    M = U @ V.T # size=(m,n)
    Omega_mask = (np.random.rand(m,n) <= p)
    Omega = matIndicesFromMask(Omega_mask)
    M_Omega = multiplyFromMatIdxList(U,V,Omega)
    return (U, V, M_Omega, Omega, Omega_mask)


def sparseMatComSetup(r,m,n,p,rng=None):
    """
    sparseMatComSetup(r,m,n,p) returns the necessary components for a simple
    set-up of a matrix completion problem.

    Input: 
      r : rank
      m : number of rows
      n : number of columns
      p : probability of uniform at random observation
    rng : the randomness to use (e.g. 
          lambda d,r : np.random.randint(5, size=(d,r))
          lambda d,r : 2*np.random.randn(d,r)

    Output:
          U : The left [sparse] matrix such that U @ V.T = M
          V : The right [sparse] matrix V such that U @ V.T = M
      Omega : A tuple (I, J) containing a vector of row and column indices
              corresponding to which entries of M were observed.
        obs : A vector of observations corresponding to entries of M at Omega.
    M_Omega : An m-by-n sparse matrix s.t. M_Omega[Omega] == obs.
    """
    if rng is None:
        rng = lambda d,r: np.random.randint(5, size=(d,r))
    k = np.random.binomial(m*n, p)
    Omega = (np.random.randint(m, size=k), np.random.randint(n, size=k))
    U = rng(m,r)
    V = rng(n,r)
    observations = multiplyFromMatIdxList(U, V, Omega)
    M_Omega = csr_matrix((observations, Omega), 
                                shape=(m,n))
    return (U, V, Omega, observations, M_Omega)


def multiplyFromMatIdxList(U, V, Omega):
    """
    multiplyFromMatIdxList(U, V, Omega) returns a vector M_Omega 
    where each entry is given by 
        M_jk := < U_j, V_k >, for (j,k) \in Omega

    Input:
        U : The m-by-r left low-rank matrix
        V : The n-by-r right low-rank matrix
    Omega : A tuple of vectors, the first representing a list of 
            row indices, the second column indices. The tuple formed
            by the i-th element of each vector corresponds to an
            observed element of the low-rank matrix U @ V.T
    """
    return np.array([U[j,:] @ V[k,:] for j,k in zip(*Omega)])


def plot_error(M, M_rec, **kwargs):
    """
    plot_error(M, M_rec, **kwargs)
    """
    figsize=kwargs.get('figsize', (6,6))
    cmap = kwargs.get('cmap', 'viridis')
    fontsize = kwargs.get('size', 16)
    error_map = kwargs.get('error_map', lambda x,y: np.abs(x-y))
    ptitle = kwargs.get('error_title', 'absolute error $|M - M^*|$')
    cb = kwargs.get('show_colorbar', True)
    
    error = error_map(M, M_rec)
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    I = ax.matshow(error, cmap=cmap)
    ax.set_title(ptitle, size=fontsize)
    ax.axis('off')

    if cb:
        cb_orient = kwargs.get('orientation', 'vertical')
        cb_pad = kwargs.get('pad', 0.05)
        fig.colorbar(I, ax=ax, orientation=cb_orient, pad=cb_pad)

    print('RMSE = {}'.format(rmse(M, M_rec)))
    return

    
def plot_cplx(cplx, **kwargs):
    """
    plot_cplx(cplx, **kwargs) is a simple function to plot the real 
    and imaginary components of a complex-valued array.

    Input
    cplx : the complex-valued array
    figsize : the figure size (default: (12,6))
    cmap : the color map ot use (default: gray)
    """
    # options
    figsize = kwargs.get('figsize', (12,6))
    cmap = kwargs.get('cmap', 'gray')
    cb = kwargs.get('cb', True)
    cb_tied = kwargs.get('cb_tied', True)

    vmin_real, vmax_real = cplx.real.min(), cplx.real.max()
    vmin_imag, vmax_imag = cplx.imag.min(), cplx.imag.max()
    if cb and cb_tied:
        vmin = [np.minimum(vmin_real, vmin_imag)]*2
        vmax = [np.maximum(vmax_real, vmax_imag)]*2
    else:
        vmin = [vmin_real, vmin_imag]
        vmax = [vmax_real, vmax_imag]
    
    #plots
    fig, axes = plt.subplots(1,2,figsize=figsize)
    im0 = axes.flat[0].matshow(cplx.real, cmap=cmap,
                              vmin=vmin[0], vmax=vmax[0])
    axes.flat[0].axis('off')
    im1 = axes.flat[1].matshow(cplx.imag, cmap=cmap,
                               vmin=vmin[1], vmax=vmax[1])
    axes.flat[1].axis('off')

    if cb and cb_tied:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im0, cax=cbar_ax)
    elif cb:
        fig.colorbar(im0, ax=axes.flat[0])
        fig.colorbar(im1, ax=axes.flat[1])
    return


def plot_comparison(M, M_rec, show_error=True, **kwargs):
    """
    plot_comparison(M, M_rec, show_error=True, show_colorbars=True, **kwargs)
        plots two matrices, M and M_rec, side-by-side along with
        optionally a plot of the error between the two (given by some
        error map error_map)
    Input:
    M : the original matrix M
    M_rec : the recovered matrix M_rec
    show_error : if True (default) produce a third plot showing a 
                 mapping of (M, M_rec) 
                 (default: absolute error |M-M_rec|)
    show_colorbars : 
    kwargs:
    figsize : the figure size (w, h); default is (15,8)
    cmap : a string denoting the colormap to use for the plots,
           defined according to plt.cm.cmaps_lsited (default: 'viridis')
    fontsize : the font size for the labels on the plot (default: 16)
    plot_titles : the plot titles for the first two plots 
                  (default: ['$M$', '$M^*$'])
    error_map : the error map to use for the third plot 
                (only used if show_error=True). Default value is absolute
                error, |M - M_rec|.
    error_title : the title to use for the third plot. 
    cb : if True (default) plot colorbars beneath the plots (recommended)
    cb_orient : whether to show 'horizontal' or 'vertical' orientation 
                of colourbar (default horizontal)
    cb_pad : padding between plot and colourbar (default .05)
    """
    n_plots = 2
    if show_error:
        n_plots += 1
    figsize=kwargs.get('figsize', (15,8))
    cmap = kwargs.get('cmap', 'viridis')
    fontsize = kwargs.get('size', 16)
    plot_titles = kwargs.get('plot_titles', ['$M$', '$M^*$'])
    error_map = kwargs.get('error_map', lambda x,y: np.abs(x-y))
    error_title = kwargs.get('error_title', 'absolute error $|M - M^*|$')
    cb = kwargs.get('show_colorbars', True)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    matrices = [M, M_rec]
    if n_plots == 3:
        matrices.append(error_map(M, M_rec))
        plot_titles.append(error_title)
    for ax, mat, ptitle in zip(axes.flat, matrices, plot_titles):
        I = ax.matshow(mat, cmap=cmap)
        ax.set_title(ptitle, size=fontsize)
        ax.axis('off')
        if cb:
            cb_orient = kwargs.get('orientation', 'horizontal')
            cb_pad = kwargs.get('pad', 0.05)
            fig.colorbar(I, ax=ax, orientation=cb_orient, pad=cb_pad)
    print('RMSE = {}'.format(rmse(M, M_rec)))
    return


def vec(A, stack="columns"):
    """
    vec(A) returns the vectorization of the matrix A
    by stacking the columns (or rows, respectively) of A.
    """
    if stack[0].lower() == 'c':
        return A.T.ravel()
    elif stack[0].lower() == 'r':
        return A.ravel()
    else:
        raise ValueError('Expected \'columns\' or \'rows\' for argument stack.')


def unvec(vecA, shape):
    """
    _unvec(A, shape) returns the "unvectorization" of the
    matrix A by unstacking the columns of vecA to return
    the matrix A of shape shape.
    """
    return vecA.reshape(shape, order='F')


def vecIndicesFromMask(mask, stack='columns'):
    """
    vecIndicesFromMask(mask, stack='columns')
    returns the vector-indices corresponding to mask == 1.
    This is operation is performed by first vectorizing the
    mask array.
    """
    return np.where(vec(mask, stack)==1)[0]

def matIndicesFromMask(mask):
    """
    matIndicesFromMask(mask) returns the matrix-indices 
    corresponding to mask == 1. This operation returns a 
    tuple containing a list of row indices and a list of 
    column indices.
    """
    return np.where(mask.T==1)[::-1]


def masked(A, mask):
    """
    masked(A, mask) returns the "observed entries" of the
    matrix A, as a vector, determined according to the 
    condition mask == 1 (alternatively, the entries for 
    which mask is True).
    """
    return A[matIndicesFromMask(mask)]


def _get_sparse_type(st=None):
    """
    _get_sparse_type(st=None) is a bookeeping function to that determines 
    which type of sparse matrix to return, given its argument st.
    Note: st can be a function (e.g. scipy.sparse.csr_matrix), a string 
    (e.g., 'csr', 'csr_matrix'), or None (returns scipy.sparse.csr_matrix). 
    The output of this function is the corresponding sparse matrix constructor
    (e.g., scipy.sparse.csr_matrix). 
    """
    if (st is None) or (isinstance(st, 'str') and (st[:3] == 'csr')):
        from scipy.sparse import csr_matrix
        return csr_matrix
    elif isinstance(st, 'str') and (st[:3] == 'csc'):
        from scipy.sparse import csc_matrix
        return csc_matrix
    elif isinstance(st, 'str') and (st[:3] == 'coo'):
        from scipy.sparse import coo_matrix
        return coo_matrix
    elif isinstance(st, 'str') and (st[:3] == 'dok'):
        from scipy.sparse import dok_matrix
        return dok_matrix
    elif isinstance(st, 'str') and (st[:3] == 'lil'):
        from scipy.sparse import lil_matrix
        return lil_matrix
    else:
        raise ValueError('Could not detect type of sparse matrix constructor to return.')


def matricize_right(V, Omega, m=None, sparse=True, sparse_type=None):
    """
    matricize_right(V, Omega, m=None, sparse=True, sparse_type=None) 
    turns the problem 
        M_Omega = (U @ V.T)_Omega 
    into the matrix problem
        vec(M_Omega) = W @ vec(U)
    where U is an m-by-r matrix, V is an n-by-r matrix and
        vec([[1,2,3],[4,5,6]]) = [1,4,2,5,3,6].T

    Input
              V : the right n-by-r matrix
          Omega : the mask / list of indices of observed entries
         sparse : whether to return a sparse matrix (default: true)
    sparse_type : what kind of sparse matrix to return (default: csr)

    Output
    V_op : The operator for V in matrix form so that vec(U @ V.T) is 
           equivalent to V_op @ vec(U).
    """
    if isinstance(Omega, tuple):
        Omega_i = Omega[0]
        Omega_j = Omega[1]
        if m is None:
            raise ValueError('input number of columns for left' +
                             ' factor is required when Omega is a ' +
                             'list of indices')
    elif isinstance(Omega, np.ndarray):
        m = Omega.shape[0]
        Omega_i, Omega_j = matIndicesFromMask(Omega)
    else:
        raise ValueError('type of Omega not recognized; ' + 
                         'expected tuple of indices or mask array.')
    r = V.shape[1]
    sizeU = m*r
    if sparse:
        sp_mat = _get_sparse_type(sparse_type)
        row_idx = np.repeat(range(Omega_i.size), r)
        col_idx = [np.arange(Omega_i[n], sizeU, m, dtype=int) 
                   for n in range(Omega_i.size)]
        col_idx = np.concatenate(col_idx)
        vals = np.concatenate([V[j,:] for j in Omega_j])
        V_op = sp_mat((vals, (row_idx, col_idx)), shape=(Omega_i.size, sizeU))
    else:
        V_op = np.zeros((Omega_i.size, sizeU))
        for n in range(Omega_i.size):
            i = Omega_i[n]
            j = Omega_j[n]
            V_op[n, i::m] = V[j,:]
    return V_op


def matricize_left(U, Omega, n=None, sparse=True, sparse_type=None):
    """
    matricize_left(U, Omega, n=None, sparse=True, sparse_type=None) 
    turns the problem
        M_Omega = (U @ V.T)_Omega
    into the matrix problem
        vec(M_Omega) = W @ vec(V)
    where U is an m-by-r matrix, V is an n-by-r matrix and
        vec([[1,2,3],[4,5,6]]) = [1,4,2,5,3,6].T

    Input
              U : the left m-by-r matrix
          Omega : the mask / list of indices of observed entries
         sparse : whether to return a sparse matrix (default: true)
    sparse_type : what kind of sparse matrix to return (default: csr)

    Output
    U_op : The operator for U in matrix form so that vec(U @ V.T) is 
           equivalent to U_op @ vec(V).
    """
    if isinstance(Omega, tuple):
        Omega_i = Omega[0]
        Omega_j = Omega[1]
        if n is None:
            raise ValueError('input number of columns for right' +
                             ' factor is required when Omega is a ' +
                             'list of indices')
    elif isinstance(Omega, np.ndarray):
        n = Omega.shape[1]
        Omega_i, Omega_j = matIndicesFromMask(Omega)
    else:
        raise ValueError('type of Omega not recognized; ' + 
                         'expected tuple of indices or mask array.')

    r = U.shape[1]
    sizeV = n*r

    if sparse:
        sp_mat = _get_sparse_type(sparse_type)
        row_idx = np.repeat(range(Omega_j.size), r)
        col_idx = [np.arange(Omega_j[idx], sizeV, n, dtype=int) 
                   for idx in range(Omega_j.size)]
        col_idx = np.concatenate(col_idx)
        vals = np.concatenate([U[i,:] for i in Omega_i])
        U_op = sp_mat((vals, (row_idx, col_idx)), shape=(Omega_j.size, sizeV))
    else:
        U_op = np.zeros((Omega_j.size, sizeV))
        for idx in range(Omega_j.size):
            i = Omega_i[idx]
            j = Omega_j[idx]
            U_op[idx, j::n] = U[i,:]
    return U_op



def load_seismic_data(directory='./data/seismic/', 
                      csv='seismic.csv', 
                      readme='SeismicData.md'):
    """
    load_seismic_data(directory='./data/seismic/', 
                      csv='seismic.csv', readme='SeismicData.md')
    is a function to load in the seismic data for the matrix 
    completion mini-project. Default parameter values are shown 
    above. If a path to a README file is passed, then this function
    will also print out the associated README (to stdout in Terminal
    or in fancy markdown format in Jupyter). 

    Input
    directory : the directory where the seismic data is located
    csv : the csv file containing the data
    readme : the readme file associated to the data
             (if there is no readme file, set readme=None)

    Output
    seismic : The complex-valued seismic array.
    """
    # load seismic data
    seismic = np.loadtxt(directory+csv, dtype=np.complex, delimiter=',')

    # handle readme file
    if readme is not None:
        with open(directory+readme, 'r', encoding='utf-8') as fp:
            readme = fp.read()
        try:
            curr_instance = get_ipython()
            from IPython.display import display, Markdown
            display(Markdown(readme))
        except:
            print(readme)

    # return!
    return seismic


def load_movie_ratings(directory='./data/movielens-latest/', readme='README.txt'):
    """
    load_movie_ratings(directory='./data/movielens-latest/',
                       links='links.csv', movies='movies.csv',
                       ratings='ratings.csv', tags='tags.csv',
                       readme='README.txt'):
    is a function to load in the movie lens data for the matrix 
    completion mini-project. Default parameter values are shown 
    above. If a path to a README file is passed, then this function
    will also print out the associated README (to stdout in Terminal
    or in fancy markdown format in Jupyter). 
    This function expects there to exist the following filenames in directory:
    "links.csv", "movies.csv", "ratings.csv", "tags.csv". 

    Input
    directory : the directory where the data is located
    readme : the readme file associated to the data
             (if there is no readme file, set readme=None)

    Output
    links : 
    movies : 
    ratings : 
    tags : 
    """
    filenames = [x + '.csv' for x in ['links', 'movies', 'ratings', 'tags']]
    dfFromCsv = pd.DataFrame.from_csv
    movie_ratings = (dfFromCsv(directory + x) for x in filenames)
    # handle readme file
    if readme is not None:
        with open(directory+readme, 'r', encoding='utf-8') as fp:
            readme = fp.read()
        try:
            curr_instance = get_ipython()
            from IPython.display import display, Markdown
            display(Markdown(readme))
        except:
            print(readme)

    # return!
    return movie_ratings



def seismic_mask(seismic, p=.5, seed=None):
    """
    seismic_mask(seismic, p=.5, seed=None) computes a mask for 
    a complex seismic array, where observations are observed 
    uniformly at random with probability p. 
    
    Input
    seismic : 
    p : 
    seed : 

    Output:
    maskArr : 
    maskIdx : 
    seismicObs : 
    """
    if seed is not None:
        np.random.seed(seed)
    maskArr = (np.random.rand(*seismic.shape) < p)
    maskIdx = matIndicesFromMask(maskArr)
    seismicObs = seismic[maskIdx]
    return maskArr, maskIdx, seismicObs




def aNonuniformSampling(m, n, p=None, k=None, 
                        rowCutoff=None, colCutoff=None):
    """
    aNonuniformSampling(m, n, k, p, rowCutoff, colCutoff)
    returns a tuple (i, j) corresponding to a collection of 
    indices (ii, jj) \in [m]x[n]. Specifically, for ease of 
    implementation and demonstration purposes, each index (ii, jj)
    actually lies either in [rowCutoff] x [colCutoff] or in
    {rowCutoff, ..., m-1}x{colCutoff, ..., n-1}. 
    
    Input
    m, n: the maximum number of rows and columns, respectively.
    k: the total number of indices to return.
    p: parameter controlling the undersampling of the right-bottom
        region relative to the top-left region.
    rowCutoff, colCutoff: outline the bounds of the top-left vs. 
                          bottom-right regions of the matrix.
    """
    # error handling
    if (k is None) or (k < 1) or (k > m*n):
        k = np.int(.1 * m*n)
    if (rowCutoff is None) or (rowCutoff < 0) or (rowCutoff > m-1): 
        rowCutoff = np.int(m/2)
    if (colCutoff is None) or (colCutoff < 0) or (colCutoff > n-1):
        colCutoff = np.int(n/2)
    if (p is None) or (p < 0) or (p > 1):
        p = .1

    # sample row, col indices for upper-left
    i1, j1 = (np.random.randint(rowCutoff, size=k), np.random.randint(colCutoff, size=k))
    # sample row, col indices for lower-right
    i2, j2 = (np.random.randint(rowCutoff, m, size=k), np.random.randint(colCutoff, n, size=k))
    # the mask will determine for which indices top-left "wins"
    mask = (np.random.rand(k) > p).astype(int)
    
    # prepare and return the result
    i = mask * i1 + (1-mask) * i2
    j = mask * j1 + (1-mask) * j2
    return (i,j)
