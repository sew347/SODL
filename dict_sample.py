import numpy as np
import random
import math
import pickle
import argparse
import warnings
import time
import os
import numpy.linalg as la

#dict_sample.py
#### inputs ####
#M = data dimension
#s = sparsity
#K = number of dictionary elements
#N = number of samples
#D = optional input dictionary
#testmode = TO BE REMOVED
#bias_weight = weight for biased supp_distrib probability
#epsi = noise strength (not thoroughly tested)
#lowmem = if set, calculate thresholding in blocks. Semi-deprecated
# thresh = thresholding parameter
# fixed_supp = list of fixed dictionary element in first samples; e.g. fixed_supp = [1,5,7,0] means the first sample will always contain dictionary element 1 in its support, the second contains 5, the third 7, and the fourth 0.
# n_subspaces = parameter indicating how many subspaces to recover in support recovery. Used to limit the number of inner products that must be computed and stored.
#full_corr = TO BE REMOVED
#### fields ####
#other than above inputs,
#D = dictionary of uniformly random M-dimensional vectors
#X = sparsity pattern matrix; each column is s-sparse.
#Y = sample data D*X; +noise if epsi > 0
#HSig_D = Y*Y^T/N
#corr = inner products of first n_subspaces columns of Y with all elements in Y
######################################################

class dict_sample:
    def __init__(self, M, s, K, N, D = None, epsi = 0, fixed_supp = [], n_subspaces = -1, use_complex = False):
        self.M = M
        self.s = s
        self.K = K
        self.N = N
        self.fixed_supp = fixed_supp
        self.n_subspaces = self.N if n_subspaces == -1 else n_subspaces
        self.use_complex = use_complex
        self.D = self.build_D()
        self.X = self.build_X()
        self.Y = self.build_Y()
        if epsi > 0:
            print('Test suite is running with noise.')
            self.Y = self.Y + np.random.normal(0,epsi*math.sqrt(s)/math.sqrt(M),(M,N))
        self.HSig_D = self.build_HSig_D()
        self.corr = np.abs(np.dot(np.transpose(self.Y[:,:self.n_subspaces]),self.Y))

    def build_D(self):
        D = np.random.normal(0,1,(self.M,self.K))
        if self.use_complex:
            D = D.astype(np.complex128)
            D = D + np.random.normal(0,1,(self.M,self.K))*1j
        D = D/np.linalg.norm(D, axis = 0)
        return(D)

    def build_X(self):
        X = np.zeros((self.K,self.N))
        for i in range(self.N):
            X[:,i] = self.get_Xcol(i)
        return(X)

    def build_Y(self):
        return np.dot(self.D,self.X)

    def get_Xcol(self,i):
        Xcol = np.zeros(self.K)
        if i < len(self.fixed_supp):
            supp_elem = self.fixed_supp[i]
            remaining_rows = list(range(supp_elem))+list(range(supp_elem+1,self.K))
            rows = [supp_elem] + random.sample(remaining_rows,self.s-1)
            Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
        else:
            rows = random.sample(range(self.K),self.s)
            Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
        return Xcol

    def build_HSig_D(self):
        HSig_D = np.dot(self.Y, np.transpose(self.Y))
        return HSig_D/np.linalg.norm(HSig_D)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script creates a random dictionary and samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--M', type=int, help='Sample dimension', required=True)
    parser.add_argument('--s', type=int, help='Sparsity', required=True)
    parser.add_argument('--K', type=int, help='Size of dictionary', required = False)
    parser.add_argument('--N', type=int, help='Number of samples', required = False)
    parser.add_argument('--seed', type=int, help='Random seed', required = False)
    parser.add_argument('--fixed_supp', type=int, nargs='+', help='fixed support', required=False, default = [])
    parser.add_argument('--use_complex', help='If true, use complex numbers', required=False, action='store_true')
    parser.add_argument('--output_folder', help='Folder for saving output', required = True)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    data = dict_sample(args.M,args.s,args.K,args.N, fixed_supp = args.fixed_supp, use_complex = args.use_complex)
    X = data.X
    D = data.D
    Y = data.Y
    path = args.output_folder
    
    if not os.path.exists(path):
       os.makedirs(path)
    
    np.save(path+'/X.npy', X)
    np.save(path+'/D.npy', D)
    np.save(path+'/Y.npy', Y)