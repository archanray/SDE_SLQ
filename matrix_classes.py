import numpy as np
import numpy.polynomial as poly
from scipy import sparse
from scipy import integrate
import scipy.sparse.linalg
import math 
from collections import deque
import pandas as pd
import pickle
import time


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









class MatrixObjectDense: 
	def __init__(self, mat_filename): 
		self.filename = mat_filename
		self.mat = load_pickle(self.filename)
		self.n = np.shape(self.mat)[0]
		self.col_norms_sqd = np.linalg.norm(self.mat, axis=0)**2
		self.frob_sqd = np.sum(self.col_norms_sqd)
		self.col_probs = self.col_norms_sqd/np.sum(self.col_norms_sqd)
		self.nnz = np.count_nonzero(self.mat)

	def normalize_mat(mat, op_norm): 
		return mat/(2*op_norm)

	def create_simple_matmul_fn(self):
		# one_mat = (self.mat != 0.).astype(float)
		def fn(mat_like): 
			ret = np.zeros((self.n, mat_like.shape[1]))
			matmul_sp = 0.0
			for j in range(mat_like.shape[1]):
				vec = mat_like[:, j]
				# matmul_sp += np.sum(np.matmul(one_mat, (vec.astype(bool)).astype(float)))			
			return (np.matmul(self.mat, mat_like), matmul_sp/mat_like.shape[1])
		return fn
	  
	# def create_column_matmul_fn(self, col_budget): 
	# 	def column_matmul(mat_like): 
	# 		ret = np.zeros((self.n, mat_like.shape[1]))
	# 		for j in range(mat_like.shape[1]):
	# 			s = np.random.choice(np.arange(0, (self.mat).shape[1], 1), size=col_budget, p=self.col_probs)
	# 			mat_sliced_normalized = (self.mat[:, s])*(1./(self.col_norms_sqd[s]*col_budget/self.frob_sqd))
	# 			ret[:, j] = np.matmul(mat_sliced_normalized, mat_like[s, j])
	# 		return ret
	# 	return column_matmul

	# def create_column_matmul_fn(self, col_budget): 
	# 	def column_matmul(mat_like): 
	# 		ret = np.zeros((self.n, mat_like.shape[1]))
	# 		matmul_sp = 0.0
	# 		for j in range(mat_like.shape[1]):
	# 			s = np.random.choice(np.arange(0, (self.mat).shape[1], 1), size=col_budget, p=self.col_probs)				
	# 			su, counts = np.unique(s, return_counts=True)
	# 			vec = np.zeros(mat_like.shape[0])
	# 			vec[su] = mat_like[su, j]*(counts/(self.col_probs[su]*col_budget))
				
	# 			ret[:, j] = np.matmul(self.mat, vec)
	# 			# matmul_sp += np.count_nonzero(np.matmul(self.mat, np.diag(vec)))
	# 		return (ret, matmul_sp/mat_like.shape[1])
	# 	return column_matmul

	def create_column_matmul_fn(self, oversample_factor): 
		# one_mat = (self.mat != 0.).astype(float)
		def column_matmul(mat_like): 
			ret = np.zeros((self.n, mat_like.shape[1]))
			matmul_sp = 0.0
			scaled_probs = oversample_factor*self.col_probs
			for j in range(mat_like.shape[1]):				
				u = np.random.uniform(0., 1., (self.mat).shape[1])
				select_cols_vec = (((scaled_probs - u) >= 0.)).astype(float)
				vec = (select_cols_vec*(1./np.minimum(scaled_probs, 1.)))*mat_like[:, j]
				ret[:, j] = np.matmul(self.mat, vec)
				# matmul_sp += np.sum(np.matmul(one_mat, (vec.astype(bool)).astype(float)))			
			return (ret, matmul_sp/mat_like.shape[1])
		return column_matmul

	def compute_eigs(self): 
		return load_pickle(self.filename + "_eigs")








class MatrixObjectCSCSparse: 
	def __init__(self, mat_filename): 
		self.filename = mat_filename
		self.mat = scipy.sparse.csc_matrix(load_pickle(mat_filename))
		self.n = self.mat.shape[0] #scipy.shape(self.mat)[0]
		self.col_norms_sqd = scipy.sparse.linalg.norm(self.mat, axis=0)**2
		self.frob_sqd = np.sum(self.col_norms_sqd)
		self.col_probs = self.col_norms_sqd/np.sum(self.col_norms_sqd)
		self.nnz = self.mat.count_nonzero()
	# def normalize_mat(mat, op_norm): 
	# 	return mat/(2*op_norm)



	# def create_simple_matmul_fn(self):
	# 	def fn(mat_like): 
	# 		return (self.mat).dot(mat_like)
	# 	return fn

	def create_simple_matmul_fn(self):
		one_mat = (self.mat.astype(bool)).astype(float)
		def fn(mat_like): 
			ret = np.zeros((self.n, mat_like.shape[1]))
			matmul_sp = 0.0
			for j in range(mat_like.shape[1]):
				vec = mat_like[:, j]
				matmul_sp += np.sum(one_mat.dot((vec.astype(bool)).astype(float)))
			return (self.mat.dot(mat_like), matmul_sp/mat_like.shape[1])
		return fn

	
	# def create_column_matmul_fn(self, col_budget): 
	# 	def column_matmul(mat_like): 
	# 		ret = np.zeros((self.n, mat_like.shape[1]))
	# 		for j in range(mat_like.shape[1]):
	# 			s = np.random.choice(np.arange(0, (self.mat).shape[1], 1), size=col_budget, p=self.col_probs)
	# 			mat_sliced_normalized = (self.mat[:, s]).multiply(1./(self.col_norms_sqd[s]*col_budget/self.frob_sqd))
	# 			ret[:, j] = mat_sliced_normalized.dot(mat_like[s, j])
	# 		return ret
	# 	return column_matmul

	def create_column_matmul_fn(self, oversample_factor): 
		one_mat = (self.mat.astype(bool)).astype(float)
		def column_matmul(mat_like): 
			ret = np.zeros((self.n, mat_like.shape[1]))
			matmul_sp = 0.0
			scaled_probs = oversample_factor*self.col_probs
			for j in range(mat_like.shape[1]):				
				u = np.random.uniform(0., 1., (self.mat).shape[1])
				select_cols_vec = (((scaled_probs - u) >= 0.)).astype(float)
				vec = (select_cols_vec*(1./np.minimum(scaled_probs, 1.)))*mat_like[:, j]
				ret[:, j] = self.mat.dot(vec)
				matmul_sp += np.sum(one_mat.dot(vec.astype(bool).astype(float)))			
			return (ret, matmul_sp/mat_like.shape[1])
		return column_matmul

	def compute_eigs(self): 
		return load_pickle(self.filename + "_eigs")