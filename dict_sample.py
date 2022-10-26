import numpy as np
import random
import math
import pickle
import argparse
from multiprocessing import Pool, cpu_count
import warnings
import logging
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
    def __init__(self, M, s, K, N, D = None, testmode = False, epsi = 0, lowmem = False, full_corr = False, thresh = 1/2, fixed_supp = [], n_subspaces = -1):
        self.M = M
        self.s = s
        self.K = K
        self.N = N
        self.testmode = testmode
        self.fixed_supp = fixed_supp
        self.n_subspaces = self.N if n_subspaces == -1 else n_subspaces
        self.thresh = thresh
        self.lowmem = lowmem
        if N > 10**6 and not self.lowmem:
            warnings.warn("N is greater than 10^6 but lowmem mode not set. Setting lowmem automatically.")
            self.lowmem = True
        start = time.time()
        if D is not None:
            self.D = D
        else:
            self.D = self.build_D()
        self.X = self.build_X()
        self.Y = self.build_Y()
        if epsi > 0:
            print('Test suite is running with noise.')
            self.Y = self.Y + np.random.normal(0,epsi*math.sqrt(s)/math.sqrt(M),(M,N))
        else:
            self.default_thresh = 1
        self.HSig_D = self.build_HSig_D()
        if not self.lowmem:
            if full_corr:
                self.corr = np.abs(np.dot(np.transpose(self.Y),self.Y))
            else:
                self.corr = np.abs(np.dot(np.transpose(self.Y[:,:self.n_subspaces]),self.Y))
        else:
            self.uncorr_idx = self.get_corr_lowmem()

    def build_D(self):
        D = np.random.normal(0,1,(self.M,self.K))
        D = D/np.linalg.norm(D, axis = 0)
        return(D)

    def build_X(self):
        X = np.zeros((self.K,self.N))
        for i in range(self.N):
            if i == 0 and self.testmode:
                X[0:self.s,0] = np.ones(self.s)
            else:
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

    #temporary for testing deviations from uniform
    def get_Xcol_biased(self,i):
        Xcol = np.zeros(self.K)
        first_s = np.random.binomial(1,self.s/self.K,self.s)
        if np.sum(first_s) > 0:
            rows = list(np.nonzero(first_s)[0])
            n_remain = self.s - len(rows)
            inclusion_stat = self.bias_weight + np.abs(np.random.normal(0,1,self.K))
            inclusion_stat[:self.s] = 0
            largest = list(np.argpartition(inclusion_stat, -n_remain)[-n_remain:])
            rows = rows + largest
            Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
        else:
            rows = random.sample(range(self.s,self.K),self.s)
            Xcol[rows] = 1 - 2*np.random.binomial(1,0.5,self.s)
        return Xcol

    def build_HSig_D(self):
        HSig_D = np.dot(self.Y, np.transpose(self.Y))
        return HSig_D/np.linalg.norm(HSig_D)

    def reload_DY(self):
        self.D = self.build_D()
        self.Y = np.dot(self.D,self.X)
        if self.normflag:
            self.Y = self.Y/np.linalg.norm(self.Y, axis = 0)

    def get_corr_lowmem(self):
        uncorr_idx = []
        for i in range(self.N):
            uncorr_idx.append(self.get_uncorr_i(i))
        return(uncorr_idx)

    def get_uncorr_i(self, i):
        inners_i = np.dot(np.transpose(self.Y[:,i]),self.Y)
        uncorr_i = np.nonzero(np.abs(inners_i) < self.thresh)[0]
        return uncorr_i
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script creates a random dictionary and samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--M', type=int, help='Sample dimension', required=True)
    parser.add_argument('--s', type=int, help='Sparsity', required=True)
    parser.add_argument('--K', type=int, help='Size of dictionary', required = False)
    parser.add_argument('--N', type=int, help='Number of samples', required = False)
    parser.add_argument('--seed', type=int, help='Random seed', required = False)
    parser.add_argument('--fixed_supp', type=int, nargs='+', help='fixed support', required=True)
    parser.add_argument('--output_folder', help='Folder for saving output', required = True)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    data = dict_sample(args.M,args.s,args.K,args.N, fixed_supp = args.fixed_supp)
    X = data.X
    D = data.D
    Y = data.Y
    path = args.output_folder
    
    if not os.path.exists(path):
       os.makedirs(path)
    
    np.save(path+'/X.npy', X)
    np.save(path+'/D.npy', D)
    np.save(path+'/Y.npy', Y)