import numpy as np

#Input: Sample Y, index i, sparsity s

def recover_subspace(i, s, Y, cov, corrs = None):
    if corrs is None:
        corrs = Y[:,i].T @ Y
    mu = np.mean(np.abs(corrs))
    corr_weight_samples = (corrs/mu) * Y
    N = np.shape(Y)[1]
    corr_weight_cov = (corr_weight_samples @ corr_weight_samples.T)/N
    cwc_proj = frobenius_projection(corr_weight_cov, cov)
    M = np.shape(cwc_proj)[0]
    E = np.linalg.eigh(cwc_proj)
    return E[1][:,-s:]


def frobenius_projection(A, B):
    inner_prod = np.trace(A @ B)
    return A - inner_prod*B