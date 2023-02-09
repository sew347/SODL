import numpy as np
import argparse
import os
import single_subspace_recovery as ssr
import scipy.io as sio


def recover_subspaces(Y, s, output_folder, J=None):
    """
    Recovers bases for the spanning subspaces from sample matrix Y.
    :param Y: Data matrix with samples as columns
    :param s: Expected sparsity
    :param output_folder: Subspaces will be saved in this location
    :param J: Number of subspaces to recover. If not set, all will be recovered.
    :return: None
    """
    if J is None:
        J = np.shape(Y)[1]
    N = np.shape(Y)[1]
    cov = (Y @ Y.T)/N
    corrs = Y[:,:J].T @ Y
    for i in range(J):
        subspace_curr = ssr.recover_subspace(i, s, Y, cov, corrs = corrs[i,:])
        filename = output_folder + '/subspace_' + str(i) +'.npy'
        np.save(filename, subspace_curr)
        if np.mod(i+1, 50) == 0:
            print(f'Computed {i+1} subspaces successfully.')

def recover_subspaces_from_file(sample_file, s, output_folder, J = None, matlab_name = 'Y'):
    """
    Wrapper for recover_subspaces to be called from file for samples.
    :param sample_file: location of file in .npy or .mat format containing matrix with samples as columns
    :param s: Expected sparsity
    :param J: Number of subspaces to recover. If not set, all will be recovered.
    :param output_folder: If set, subspaces will be saved in this location
    :param matlab_name: Variable name of matlab element to load as Y matrix. Defaults to 'Y'.
    :return: List of recovered subspaces
    """
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
    return recover_subspaces(Y, s, output_folder, J = J)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of sample vectors and returns a collection of estimated spanning subspaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sample_file', help='Location of sample file', required=True)
    parser.add_argument('--s', type=int, help='Expected sparsity', required=True)
    parser.add_argument('--output_folder', help='Folder for saving output', required = True)
    parser.add_argument('--J', type=int, help='Number of subspaces to compute', required = False)
    parser.add_argument('--matlab_name', type=str, help='Variable name for .mat files', default = 'Y', required = False)
    args = parser.parse_args()
    
    path = args.output_folder
    if not os.path.exists(path):
       os.makedirs(path)

    recover_subspaces_from_file(
        args.sample_file,
        args.s,
        args.output_folder,
        J = args.J,
        matlab_name = args.matlab_name,
    )

