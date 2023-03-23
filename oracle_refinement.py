import numpy as np
import utils
import os
import argparse
import pickle
import pickle


def check_support(ora, subspace_proj, thresh=1/2):
    resid = ora - subspace_proj @ ora
    if np.linalg.norm(resid) < thresh:
        return True
    else:
        return False

def oracle_support(est_D, subspaces, s = None, thresh=1/2):
    N = len(subspaces)
    K = est_D.shape[1]
    support = [[] for i in range(N)]
    has_k = [[] for i in range(K)]
    for i in range(N):
        n_supp = [0]*N
        P_i = subspaces[i] @ np.conjugate(subspaces[i]).T
        for k in range(K):
            if check_support(est_D[:,k],P_i):
                support[i].append(k)
                has_k[k].append(i)
                if s is not None:
                    n_supp[i] += 1
                    if n_supp[i] == s:
                        break
    return support, has_k

def single_oracle_refinement(has_k, Y, cov = None):
    if cov is None:
        cov = (Y @ np.conjugate(Y).T)/Y.shape[1]
    
    oracle_cov = (Y[:,has_k] @ np.conjugate(Y[:,has_k]).T)/len(has_k)
    oracle_cov_proj = utils.frobenius_complement(oracle_cov, cov)
    E = np.linalg.eigh(oracle_cov_proj)
    return E[1][:,-1]

def oracle_refinement(est_D, Y, subspaces, s=None, thresh=1/2):
    M = Y.shape[0]
    cov = (Y @ np.conjugate(Y).T)/Y.shape[1]
    supp, has_k = oracle_support(est_D, subspaces, s=s, thresh=thresh)
    K = len(has_k)
    ora_D = np.zeros((M,K))
    for k in range(K):
        ora_D[:,k] = single_oracle_refinement(has_k[k], Y, cov=cov) 
    return ora_D, has_k
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of sample vectors and estimated dictionary and refines it by oracle thresholding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--est_D_file', help='Filepath of existing dictionary estimate', required=True)
    parser.add_argument('--sample_file', help='Filepath of sample', required=True)
    parser.add_argument('--subspace_folder', help='Folder housing subspaces in .npy format', required=True)
    parser.add_argument('--output_folder', help='Folder for output files', required=True)
    parser.add_argument('--matlab_name', type=str, help='Variable name for .mat sample file', default = 'Y', required = False)
    parser.add_argument('--s', help='Sparsity', required=False)
    parser.add_argument('--thresh', type=float, help='Similarity threshold for estimated support', required=False, default=0.5)
    args = parser.parse_args()
    
    est_D = np.load(args.est_D_file)
    Y = utils.load_sample_file(args.sample_file, args.matlab_name)
    subspaces = utils.load_subspaces(args.subspace_folder)
    
    ora_D, has_k = oracle_refinement(est_D, Y, subspaces, s=args.s, thresh=args.thresh)
    output_folder = os.path.dirname(args.est_D_file)
    np.save(output_folder+'/oracle_D.npy', ora_D)
    with open(output_folder+'/oracle_supp.pkl', "wb") as fp:
        pickle.dump(has_k, fp)
    
    
    