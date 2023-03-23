import numpy as np
import utils
import os
import argparse
import scipy.io as sio
import pickle


def single_oracle_average(est_dk, has_k, Y):
        signs = est_dk.T @ Y[:,has_k]
        ora_avg_dk = np.sum(signs*Y[:,has_k],axis = 1)
        ora_avg_dk = ora_avg_dk/np.linalg.norm(ora_avg_dk)
        return ora_avg_dk
    

def oracle_average(ora_D, has_k, Y):
    M = Y.shape[0]
    K = len(has_k)
    ora_avg_D = np.zeros((M,K))
    for k in range(K):
        ora_avg_D[:,k] = single_oracle_average(ora_D[:,k], has_k[k], Y)
    return ora_avg_D
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of sample vectors and estimated dictionary and refines it by oracle thresholding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ora_D_file', help='Filepath of existing dictionary estimate', required=True)
    parser.add_argument('--sample_file', help='Filepath of sample', required=True)
    parser.add_argument('--support_file', help='Filepath of estimated support list', required=True)
    parser.add_argument('--matlab_name', type=str, help='Variable name for .mat files', default = 'Y', required = False)
    parser.add_argument('--s', help='Sparsity', required=False)
    args = parser.parse_args()
    
    ora_D = np.load(args.ora_D_file)
    Y = utils.load_sample_file(args.sample_file, args.matlab_name)
    with open(args.support_file,"rb") as fp:
        has_k = pickle.load(fp)
    
    ora_avg_D = oracle_average(ora_D, has_k, Y)
    np.save(os.path.dirname(args.ora_D_file)+'/ora_avg_D.npy', ora_avg_D)
    
    
    