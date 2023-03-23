import numpy as np
import numpy.linalg as la
import argparse
import pandas as pd
import os
import pdb
import utils


def proj_mat(A):
    """
    Projection matrix for orthogonal basis matrix A
    :param A: Orthogonal matrix with unit vector columns
    :return: AA^*, projection matrix onto subspace spanned by cols of A
    """
    return np.dot(A, np.conjugate(A.T))

def single_subspace_intersection(Si, Sj, tau=0.5):
    """
    Performs approximate subspace intersection on subspaces Si, Sj
    :param Si: Basis matrix for subspace
    :param Sj: Basis matrix for subspace
    :param tau: singular value threshold for intersection
    :return: Vector of intersection and Boolean whether intersection was found
    """
    Pj = proj_mat(Sj)
    P_itoj = Si - Pj @ Si
    SVD = la.svd(P_itoj)
    sing_vals = SVD[1]
    dim = get_int_dim(sing_vals, tau)
    if dim > 0:
        intersection = Si @ np.conjugate(SVD[2][-1]) if dim == 1 else Si @ np.conjugate(SVD[2][-dim:,:]).T
        return (intersection,dim)
    else:
        return (0,0)

    
def get_int_dim(sv_list, tau):
    #check if sorted:
    if not np.all(sv_list[:-1] >= sv_list[1:]):
        raise ValueError('List of singular values is not sorted.')
    for i in range(len(sv_list)):
        if sv_list[i] < tau:
            return(len(sv_list) - i)
    return 0
    

def is_new(D_list, dhat, eta):
    """
    Checks if dhat is similar to vector in D_list
    :param D_list: List of unit vectors
    :param dhat: unit vector
    :param eta: Threshold for similarity
    :return: Boolean whether |<d, dhat>|>eta for any d in D_list
    """
    is_new = True
    for d in D_list:
        # pdb.set_trace()
        if np.abs(np.inner(d,np.conjugate(dhat))) > eta:
            is_new = False
            break
    return is_new


def subspace_intersection(subspaces, D_list = [], start=0, rate=1, tau=0.5, eta=0.5):
    """
    Performs approximate subspace intersection on all pairs of subspaces Si, Sj up to J
    :param subspaces: list of bases of subspaces Si
    :param tau: singular value threshold for intersection
    :param eta: threshold for similarity
    :return: Dictionary, intersection metadata, success indicator
    """
    
    if len(D_list) > 0:
        if np.shape(D_list[0])[0] != np.shape(subspaces[0])[0]:
            raise ValueError('Existing dictionary in folder with dimension M = %d incompatible with subspace dimension M = %d'%(np.shape(D_list[0])[0], np.shape(subspaces[0])[0]))
    
    intersection_list = []
    end = len(subspaces)
    
    s = np.shape(subspaces[0])[1]
    s_remaining = [s]*end
        
    for i in range(start,end):

        Si = subspaces[i]
        if s_remaining[i] > 0:
            for j in range(i+1,end,rate):
                Sj = subspaces[j]
                int_subspace, dim = single_subspace_intersection(Si, Sj)
                if dim == 1:
                    dhat = int_subspace
                    if is_new(D_list, dhat, eta):
                        int_data = [[i,j],len(D_list), tau, eta]
                        intersection_list.append(int_data)
                        D_list.append(dhat)
                        s_remaining[i] -= 1
                        s_remaining[j] -= 1
                        if s_remaining[i] <= 0:
                            break
                            
        if i > 0 and np.mod(i+1,10) == 0:
            print(f'Completed intersections for subspace {i+1}. Recovered {len(D_list)} dictionary elements so far.')
                        
    if len(D_list) == 0:
        return False, False, False
    else:
        M = np.shape(D_list[0])[0]
        D = np.zeros((M, len(D_list))).astype(complex)
        for (i,d) in enumerate(D_list):
            D[:,i] = d
        if np.linalg.norm(D - np.real(D)) < 1e-10:
            print('Complex components of recovered dictionary below 1e-10 threshold; returning real part.')
            D = np.real(D)
    intersection_data = pd.DataFrame(intersection_list, columns = ['indices','col', 'tau', 'eta'])
    print(f'Run complete. Recovered {len(D_list)} dictionary elements.')
    return D, intersection_data, True
    

def subspace_intersection_from_files(subspace_folder, D_list = [], rate=1, tau=0.5, eta=0.5, start=None, end=None):
    """
    Performs approximate subspace intersection on all pairs of subspaces Si, Sj up to J from given directory
    :param subspace_folder: Location of saved subspaces
    :param output_folder: Location for output files
    :param tau: singular value threshold for intersection
    :param eta: threshold for similarity
    :param J: number of subspaces to scan
    :return: Dictionary, intersection metadata
    """
    subspaces = utils.load_subspaces(subspace_folder, end=end)
    D, intersection_data, success = subspace_intersection(subspaces, D_list = D_list, start=start, rate=rate, tau=tau, eta=eta)
    if success:
        return D, intersection_data
    else:
        raise ValueError('Given subspace files contained no empirical intersections with given tau.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes in a collection of estimated spanning subspaces and returns an estimated dictionary based on pairwise intersections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--subspace_folder', help='Location of input subspaces', required=True)
    parser.add_argument('--tau', type=float, help='Intersection threshold', required=False, default=0.5)
    parser.add_argument('--eta', type=float, help='Column similarity threshold', required=False, default=0.5)
    parser.add_argument('--rate', type=int, help='Subsamples intersections at this rate', required=False, default=1)
    parser.add_argument('--start', type=int, help='Start index for intersections.', required=False, default = 0)
    parser.add_argument('--end', type=int, help='End index for intersections.', required=False)
    parser.add_argument('--output_folder', help='Folder for saving output', required=True)
    args = parser.parse_args()
    
    outpath = args.output_folder
    if not os.path.exists(outpath):
       os.makedirs(outpath)
    
    if os.path.exists(outpath + '/est_D.npy'):
        input_D = np.load(outpath + '/est_D.npy')
        input_D_list = [input_D[:,i] for i in range(np.shape(input_D)[1])]
    else:
        input_D_list = []
    
    est_D, int_data = subspace_intersection_from_files(
        args.subspace_folder,
        D_list = input_D_list,
        rate = args.rate,
        tau = args.tau,
        eta = args.eta,
        start = args.start,
        end = args.end
    )
    
    if os.path.exists(outpath + '/intersection_data.csv'):
        input_metadata = pd.read_csv(outpath+'/intersection_data.csv', index_col = 0)
        int_data = pd.concat([input_metadata,int_data])
    
    np.save(outpath + '/est_D.npy', est_D)
    int_data.to_csv(outpath + '/intersection_data.csv')