import numpy as np
import argparse
import os
import single_subspace_recovery as ssr


def recover_subspaces(Y, s, J = None, output_folder = None):
    """
    Recovers bases for the spanning subspaces from sample matrix Y.
    :param Y: Data matrix with samples as columns
    :param s: Expected sparsity
    :param J: Number of subspaces to recover. If not set, all will be recovered.
    :param output_folder: If set, subspaces will be saved in this location
    :return: List of recovered subspaces
    """
    if J is None:
        J = np.shape(Y)[1]
    N = np.shape(Y)[1]
    cov = (Y @ Y.T)/N
    corrs = Y[:,:J].T @ Y
    subspaces = []
    for i in range(J):
        subspaces.append(ssr.recover_subspace(i, s, Y, cov, corrs = corrs[i,:]))
    if output_folder is not None:
        save_subspaces(output_folder, subspaces)
    return subspaces


def recover_subspaces_from_file(sample_file, s, J = None, output_folder = None):
    """
    Wrapper for recover_subspaces to be called from file for samples.
    :param sample_file: location of file in .npy format containing matrix with samples as columns
    :param s: Expected sparsity
    :param J: Number of subspaces to recover. If not set, all will be recovered.
    :param output_folder: If set, subspaces will be saved in this location
    :return: List of recovered subspaces
    """
    Y = np.load(sample_file)
    return recover_subspaces(Y, s, J, output_folder)


def save_subspaces(output_folder, subspaces):
    """
    Saves subspaces to location output_folder.
    :param output_folder: Subspaces will be saved in this location in .npy format
    :param subspaces: List of subspaces (basis matrices) to save
    """
    for (i,subspace) in enumerate(subspaces):
        filename = output_folder + '/subspace_' + str(i) +'.npy'
        np.save(filename, subspace)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of sample vectors and returns a collection of estimated spanning subspaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sample_file', help='Location of sample file', required=True)
    parser.add_argument('--s', type=int, help='Expected sparsity', required=True)
    parser.add_argument('--output_folder', help='Folder for saving output', required = True)
    parser.add_argument('--J', type=int, help='Number of subspaces to compute', required = False)
    args = parser.parse_args()
    
    path = args.output_folder
    if not os.path.exists(path):
       os.makedirs(path)

    recover_subspaces_from_file(
        args.sample_file,
        args.s,
        J = args.J,
        output_folder = args.output_folder
    )

