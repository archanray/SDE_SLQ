import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as scisp
import scipy.sparse.linalg
import sys, getopt
import os
import pickle
import matrix_classes as mat_class


def load_pickle(pickle_file):
        try:
                with open(pickle_file, 'rb') as f:
                        pickle_data = pickle.load(f)
        except UnicodeDecodeError as e:
                with open(pickle_file, 'rb') as f:
                        pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
                print('Unable to load data ', pickle_file, ':', e)
                raise
        return pickle_data


def write_pickle(data, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(data, handle)


# def return_sbmgraph_filename(num_communities, comm_size, comm_p, noncomm_p):
# 	return "data/sbm/sbm_numComm=%d_comm_size=%d_comm_p=%3.2f_noncomm_p=%3.2f" % (num_communities, comm_size, comm_p, noncomm_p)

# def return_sbmgraph_object_filename(num_communities, comm_size, comm_p, noncomm_p):
# 	return "data/sbm/sbm_object_numComm=%d_comm_size=%d_comm_p=%3.2f_noncomm_p=%3.2f" % (num_communities, comm_size, comm_p, noncomm_p)


def return_sbmgraph_filename(size, label=None):
    fn = "data/sbm/sbm_size=%d" % (size)
    if label != None: return fn + "_" + label
    return fn


def return_sbmgraph_object_filename(size, label=None):
    fn = "data/sbm/sbm_object_size=%d" % (size)
    if label != None: return fn + "_" + label
    return fn
    # return "data/sbm/sbm_object_numComm=%d_comm_size=%d_comm_p=%3.2f_noncomm_p=%3.2f" % (num_communities, comm_size, comm_p, noncomm_p)


# def random_graph_sbm(num_communities, comm_size, comm_p, noncomm_p): 
# 	n = num_communities*comm_size
# 	r = num_communities
    
# 	p = comm_p/2
# 	q = noncomm_p/2

# 	adj_mat = np.random.binomial(1, q, size=(n, n)) #Non-comm edges 
# 	for i in range(num_communities): 
# 		adj_mat[comm_size*i:comm_size*(i+1), 
# 			comm_size*i:comm_size*(i+1)] = np.random.binomial(1, p, size=(comm_size, comm_size))
# 	for i in range(n): adj_mat[i][i] = 0

# 	adj_mat = np.minimum((adj_mat + adj_mat.T), np.ones((n, n)))

# 	degs = np.linalg.norm(adj_mat, ord=1, axis=0)
# 	adj_mat = adj_mat*(1./degs)	
# 	return adj_mat


#inter_commps = r x r matrix 
def normalize_adj(adj_mat):
    degs = np.linalg.norm(adj_mat, ord=1, axis=0)
    adj_mat = adj_mat*(1./degs)	
    return np.nan_to_num(adj_mat)

def normalize_adj_sym(adj_mat):
    degs = np.linalg.norm(adj_mat, ord=1, axis=0)
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    adj_mat = np.nan_to_num(adj_mat).T
    # degs = np.linalg.norm(adj_mat, ord=1, axis=0)
    adj_mat = adj_mat*(1./np.sqrt(degs))	
    return np.nan_to_num(adj_mat)

def random_graph_sbm(args): 
    num_communities, comm_sizes, interComm_ps = args[0], args[1], args[2]
    n = int(np.sum(comm_sizes))
    # r = num_communities
    # print(n)
    adj_mat = np.zeros((n, n))
    for i in range(num_communities-1): 
        for j in range(i+1, num_communities): 
            M = np.random.binomial(1, interComm_ps[i][j], size=(comm_sizes[i], comm_sizes[j]))
            adj_mat[sum(comm_sizes[:i]):sum(comm_sizes[:i])+comm_sizes[i], 
                    sum(comm_sizes[:j]):sum(comm_sizes[:j])+comm_sizes[j]] = M

    adj_mat = np.minimum((adj_mat + adj_mat.T), np.ones((n, n)))

    for i in range(num_communities): 
        M = np.random.binomial(1, interComm_ps[i][i], size=(comm_sizes[i], comm_sizes[i]))

        adj_mat[sum(comm_sizes[:i]):sum(comm_sizes[:i])+comm_sizes[i], 
                sum(comm_sizes[:i]):sum(comm_sizes[:i])+comm_sizes[i]] = M

    for i in range(n): adj_mat[i][i] = 0.

    return normalize_adj_sym(adj_mat)


def cliquePlusMatching(args):
    n = args[0]
    assert(n%4==0)
    adj_mat = np.zeros((n, n))
    adj_mat[:n//2,:n//2] = np.ones((n//2, n//2)) - np.identity(n//2)
    adj_mat[np.arange(n//2, n, 2), np.arange(n//2+1, n, 2)] = 1.
    adj_mat[np.arange(n//2+1, n, 2), np.arange(n//2, n, 2)] = 1.

    return normalize_adj_sym(adj_mat)



def cliquePlusStar(args):
    n = args[0]
    adj_mat = np.zeros((n, n))
    adj_mat[:n//2,:n//2] = np.ones((n//2, n//2)) - np.identity(n//2)
    adj_mat[np.arange(0, n//2), np.arange(n//2, n)] = 1.
    adj_mat[np.arange(n//2, n), np.arange(0, n//2)] = 1.
    # adj_mat[np.arange(n//2+1, n, 2), np.arange(n//2, n, 2)] = 1.
    return normalize_adj_sym(adj_mat)

def hypercube_graph(args): 
    d = args[0]
    n = pow(2, d)
    adj_mat = np.zeros((n, n))
    for i in range(n): 
        for b in range(d): adj_mat[i][i ^ pow(2, b)] = 1.
    return normalize_adj_sym(adj_mat)


    




def write_sbmgraph(size, create_graph_fn, args, 
                    eigs=True, write=True, label=None, sparse=False, sym_normalize=True): 
    
    X = create_graph_fn(args)
    graphRepfn = return_sbmgraph_filename(size, label)
    write_pickle(X, graphRepfn)
    graphObjfn = return_sbmgraph_object_filename(size, label)
    
    if sparse: 
        write_pickle(mat_class.MatrixObjectCSCSparse(graphRepfn), graphObjfn)
    else: 
        write_pickle(mat_class.MatrixObjectDense(graphRepfn), graphObjfn)

    os.remove(graphRepfn)

    if eigs: 
        eigs = np.real(np.linalg.eigvals(X))
        write_pickle(eigs, graphRepfn + "_eigs")
        hist, bins = np.histogram(eigs, bins=20)
        fig, axs = plt.subplots(1, 2)
        axs[0].bar(bins[:-1], hist, width=0.1)
        # axs[1].plot(eigs, np.ones(len(eigs)), 'r|')
        plt.show()



#SBM Graphs
# num_communities = 100 #number of communites
# comm_sizes = 10*np.ones(num_communities) #Size of each community
# num_communities = 2
# comm_sizes = [500, 500]
# interComm_ps = [[0., 0.5],[0.5, 0.]]
#Edge probabilities
# interComm_ps = [[0.75, 0.25, 0.05],[0.25, 0.75, 0.15],[0.05, 0.15, 0.75]]
# interComm_ps = [[0.75, 0.75, 0.75],[0.75, 0.75, 0.75],[0.75, 0.75, 0.75]]
# interComm_ps = 0.1*(np.ones((num_communities, num_communities)) - np.identity(num_communities)) + 0.75*np.identity(num_communities)
# interComm_ps = scipy.linalg.circulant(np.arange(1., 0.25, -.75/num_communities))

# write_sbmgraph(int(sum(comm_sizes)), random_graph_sbm, [num_communities, comm_sizes.astype(int), interComm_ps])


#Clique Plus Matching Graph 
# write_sbmgraph(1000, cliquePlusMatching, [1000], label='cliquePlusMatching', sparse=True, eigs=False)


#Clique Plus Star Graph 
# write_sbmgraph(1000, cliquePlusStar, [1000], label='cliquePlusStar', sparse=True, eigs=False)

#hypercube graph 
# write_sbmgraph(pow(2, 14), hypercube_graph, [14], eigs=False, label='hypercube_graph', sparse=True)