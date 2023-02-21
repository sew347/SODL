import numpy as np

def recover_subspace(i, s, Y, cov, corrs = None, return_cov = False, cov_proj = True):
    """
    Recovers i-th subspace from samples Y
    :param i: index of subspace to recover
    :param Y: Data matrix with samples as columns
    :param s: Expected sparsity
    :param cov: covariance matrix of Y
    :param corrs: list of correlations between y_i and cols of Y
    :return: Basis for recovered susbspace
    """
    if corrs is None:
        corrs = Y[:,i].T @ np.conjugate(Y)
    abs_corrs = np.abs(corrs)
    mu = np.mean(abs_corrs)
    corr_weight_samples = (abs_corrs/mu) * Y
    N = np.shape(Y)[1]
    corr_weight_cov = (corr_weight_samples @ np.conjugate(corr_weight_samples.T))/N
    if cov_proj:
        cwc_proj = frobenius_complement(corr_weight_cov, cov)
    else:
        cwc_proj = corr_weight_cov
    M = np.shape(cwc_proj)[0]
    E = np.linalg.eigh(cwc_proj)
    if return_cov:
        return E[1][:,-s:], E, corr_weight_cov, cwc_proj
    return E[1][:,-s:]
    # return E[1][:,-s-1:-1]


def get_weighted_eigenvalues(i, Y):
    N = np.shape(Y)[1]
    cov = (Y @ Y.T)/N
    s = 1
    _, E, cwc, cwc_proj = recover_subspace(i, s, Y, cov, return_cov = True)
    return E[0]


def frobenius_complement(A, B):
    """
    Returns complement of orthogonal Frobenius projection of (symmetric) matrix A onto (symmetric) matrix B
    :param A: matrix
    :param B: matrix
    :return: A - (<A,B>_F/||B||_f^2)B
    """
    inner_prod = np.abs(np.trace(A @ np.conjugate(B)))
    return A - (inner_prod*B)/(np.linalg.norm(B,'fro')**2)