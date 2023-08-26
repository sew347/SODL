import numpy as np
import scipy.io as sio
import os

def load_sample_file(sample_file, matlab_name):
    file_ext = sample_file.split('.')[1]
    if file_ext == 'npy':
        Y = np.load(sample_file)
    elif file_ext == 'mat':
        mat_dict = sio.loadmat(sample_file)
        if matlab_name not in mat_dict.keys():
            raise ValueError('Provided matlab variable name ' + matlab_name + ' not present in file ' + sample_file + '. Confirm matlab_name parameter; default is Y.')
        Y = mat_dict[matlab_name]
    else:
        raise ValueError('File extension for sample_file must be either .npy or .mat.')
    return Y

def load_subspaces(subspace_folder, end=None):
    """
    loads subspaces into list
    """
    subspaces = []
    if not end:
        end = len(os.listdir(subspace_folder))
    for i in range(end):
        subspace_path = subspace_folder + '/subspace_' + str(i) +'.npy'
        if os.path.exists(subspace_path):
            curr_subspace = np.load(subspace_path, allow_pickle=True)
            subspaces.append(curr_subspace)
    return subspaces

def frobenius_complement(A, B):
    """
    Returns complement of orthogonal Frobenius projection of (symmetric) matrix A onto (symmetric) matrix B
    :param A: matrix
    :param B: matrix
    :return: A - (<A,B>_F/||B||_F^2)B
    """
    inner_prod = np.abs(np.trace(A @ np.conjugate(B)))
    return A - (inner_prod*B)/(np.linalg.norm(B,'fro')**2)