import numpy as np
import numpy.linalg as la
import argparse
import pandas as pd
import os


def proj_mat(A):
    return np.dot(A, np.transpose(A))

def single_subspace_intersection(Si, Sj, tau=0.5):
    Pj = proj_mat(Sj)
    P_itoj = Si - Pj @ Si
    SVD = la.svd(P_itoj)
    sing_vals = SVD[1]
    if (sing_vals[-1] < tau) and (sing_vals[-2] >= tau):
        return (Si @ SVD[2][-1],True)
    else:
        return (0,False)
    

def is_new(D_list, dhat, eta):
    is_new = True
    for d in D_list:
        if np.abs(np.inner(d,dhat)) > eta:
            is_new = False
            break
    return is_new


def subspace_intersection(subspaces, tau=0.5, eta=0.5, J=None):
    D_list = []
    intersection_list = []
    if not J:
        J = len(subspaces)

    s = np.shape(subspaces[0])[1]
    s_remaining = [s]*J
        
    for i in range(J):
        print(i)
        Si = subspaces[i]
        if s_remaining[i] > 0:
            for j in range(i+1,J):
                Sj = subspaces[j]
                dhat, has_int = single_subspace_intersection(Si, Sj)
                if has_int:
                    intersection = [i,j,-1, tau, eta]
                    if is_new(D_list, dhat, eta):
                        intersection[2] = len(D_list)
                        intersection_list.append(intersection)
                        D_list.append(dhat)
                        s_remaining[i] -= 1
                        s_remaining[j] -= 1
                        if s_remaining[i] > 0:
                            break
    if len(D_list) == 0:
        return False, False, False
    else:
        M = np.shape(D_list[0])[0]
        D = np.zeros((M, len(D_list)))
        for (i,d) in enumerate(D_list):
            D[:,i] = d
    intersection_data = pd.DataFrame(intersection_list, columns = ['idx1','idx2','col', 'tau', 'eta'])
    return D, intersection_data, True


def load_subspaces(subspace_folder, J=None):
    subspaces = []
    if not J:
        J = len(os.listdir(subspace_folder))
    for i in range(J):
        subspace_path = subspace_folder + '/subspace_' + str(i) +'.npy'
        if os.path.exists(subspace_path):
            curr_subspace = np.load(subspace_path, allow_pickle=True)
            subspaces.append(curr_subspace)
    return subspaces
    

def subspace_intersection_from_files(subspace_folder, output_folder, tau=0.5, eta=0.5, J=None):
    subspaces = load_subspaces(subspace_folder)
    D, intersection_data, success = subspace_intersection(subspaces, tau=tau, eta=eta, J=J)
    if success:
        return D, intersection_data
    else:
        raise ValueError('Given subspace files contained no empirical intersections with given tau.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of sample vectors and returns a collection of estimated spanning subspaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--subspace_folder', help='Location of input subspaces', required=True)
    parser.add_argument('--tau', type=float, help='Intersection threshold', required=False, default=0.5)
    parser.add_argument('--eta', type=float, help='Column similarity threshold', required=False, default=0.5)
    parser.add_argument('--J', type=int, help='Number of subspaces to consider.', required=False)
    parser.add_argument('--output_folder', help='Folder for saving output', required=True)
    args = parser.parse_args()
    
    outpath = args.output_folder
    if not os.path.exists(outpath):
       os.makedirs(outpath)
    
    est_D, int_data = subspace_intersection_from_files(args.subspace_folder, args.output_folder, tau = args.tau, eta = args.eta, J = args.J)
    np.save(outpath + '/est_D.npy', est_D)
    int_data.to_csv(outpath + '/intersection_data.csv')