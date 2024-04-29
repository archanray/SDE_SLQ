import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as scisp
import scipy.sparse.linalg
import sys 
import pickle

def readEdgeList_from_file(filename): 
# Takes line separated set of edges (each edge is sep with whitespace) in file
# and computes a dictionary rep of the graph [node]:[list of adjacent nodes].
# Dumps object in file.
	gf = open(filename)
	g = {}
	for l in gf.readlines():
		l = l.strip()
		if len(l) > 0: 
			a = map(int, l.split())
			if a[0] not in g: g[a[0]] = [a[1]]
			else: g[a[0]] += [a[1]]		
			if a[1] not in g: g[a[1]] = [a[0]]
			else: g[a[1]] += [a[0]]			
	#Need to renumber nodes so that they aren't skipped. 
	i = 0
	v_map = {}
	for v in g.keys(): 
		v_map[v] = i 
		i+=1
	g_new = {}
	for v in g.keys(): 
		g_new[v_map[v]] = map(lambda u : v_map[u], g[v])
	return g_new





# def write_to_file(g, filename):	
# # Takes in graph (of some representation) and writes to 
# # a file with filename
# 	output_fn = open(filename, "w+")
# 	pickle.dump(g, output_fn)

# def load_from_file(filename): 
# # Loads graph from file with filename
# 	gf = open(filename)
# 	g = pickle.load(gf)
# 	return g

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


def return_smbgraph_filename(num_communities, comm_size, comm_p, noncomm_p):
	return "sbm_numComm=%d_comm_size=%d_comm_p=%3.2f_noncomm_p=%3.2f" % (num_communities, comm_size, comm_p, noncomm_p)

def write_graphRep(fn):
#Takes input file name that contains list of edges
#computes graph dictionary and writes to folder 
	g_dict = readEdgeList_from_file(fn)
	write_to_file(g_dict, fn.strip('.txt') + "_graphRep")

def write_normAdjMat_sparse(graphRepfn):
	graph = load_pickle(graphRepfn)
	nodes = graph.keys()
	n = max(nodes) + 1
	print ("n=%d" % n)
	adj_mat = scipy.sparse.csc_matrix((n, n))
	for v in nodes:
		adj_mat[v, graph[v]] = 1.

	degs = scipy.sparse.linalg.norm(adj_mat, ord=1, axis=0)
	normAdjMat = adj_mat.multiply(1./degs)

	eigs = scipy.sparse.linalg.eigs(normAdjMat, k=n-2, return_eigenvectors=False)
	write_pickle(eigs, graphRepfn + "_normAdjMat_eigs")
	plt.plot(eigs, np.ones(len(eigs)), 'r+')
	plt.show()
	write_pickle(normAdjMat, graphRepfn + "_normAdjMat")


# X = np.outer(np.arange(1, 11, 1), np.arange(1, 11, 1))
# X = scipy.sparse.csc_matrix(X)
# X = np.array(X.todense())
# print (X* ((1./(np.linalg.norm(X, ord=1, axis=0)))))
# print ((X.multiply(1./np.arange(10))).todense())

# write_sbmgraph(10, 100, 0.5, 0.01)
write_normAdjMat_sparse("ca-GrQc/ca-GrQc_graphRep")


