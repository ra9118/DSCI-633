import pandas as pd
import numpy as np
from collections import Counter

class my_DT:

	def __init__(self,
				 max_depth=8,
				 min_im_y_decrease=0,
				 min_samples_split=2):

		self.max_depth = int(max_depth)
		self.min_im_y_decrease = min_im_y_decrease
		self.min_samples_split = int(min_samples_split)
		self.tree = {}

	def gini(self, labels):

		s = Counter(labels)
		N = float(len(labels))
		i_o = 1
		for key in s:
			i_o -= (s[key] / N) ** 2

		return i_o

	def find_node(self,i_s,i_o,cans,labels,cs,pop,n):
				for i in range(n - 1):
					i_s.append([] if cans[cs[i]] == cans[cs[i + 1]] else [self.gini(labels[pop[cs[:i + 1]]]) * (i + 1), (n - i - 1) * self.gini(labels[pop[cs[i + 1:]]])])
					i_o.append(np.inf if cans[cs[i]] == cans[cs[i + 1]] else np.sum(i_s[-1]))
				return [i_s,i_o ]

	def find_best_split(self, pop, X, labels):

		s_f = None
		for feature in X.keys():
			cans = np.array(X[feature][pop])
			cs = np.argsort(cans)
			n = len(cs)
			[i_s,i_o ] = self.find_node([],[],cans,labels,cs,pop,n)
			min_i_o = np.min(i_o)

			if min_i_o < np.inf and (s_f == None or s_f[1] > min_i_o):
				split = np.argmin(i_o)
				s_f = (feature, min_i_o, (cans[cs][split] + cans[cs][split + 1]) / 2.0,
								[pop[cs[:split + 1]], pop[cs[split + 1:]]], i_s[split])

		return s_f

	def fit_node(self,nodes,labels,n_nodes,p,im_y,n_level,X,N):
		for node in nodes:
			left_n = node * 2 + 1
			right_n = node * 2 + 2
			c_pop = p[node]
			c_i_o = im_y[node]
			if len(c_pop) < self.min_samples_split or c_i_o == 0 or n_level:
				self.tree[node] = Counter(labels[c_pop])
			else:
				s_f = self.find_best_split(c_pop, X, labels)
				if s_f and (c_i_o - s_f[1]) > self.min_im_y_decrease * N:
					self.tree[node] = (s_f[0], s_f[2])
					n_nodes.extend([left_n, right_n])
					p[left_n] = s_f[3][0]
					p[right_n] = s_f[3][1]
					im_y[left_n] = s_f[4][0]
					im_y[right_n] = s_f[4][1]
				else:
					self.tree[node] = Counter(labels[c_pop])
		return n_nodes

	def fit(self, X, y):

		self.c_s = list(set(list(y)))
		labels = np.array(y)
		N = len(y)
		p = {0: np.array(range(N))}
		im_y = {0: self.gini(labels[p[0]])*N}
		level = 0
		nodes = [0]
		while level < self.max_depth and nodes:
			nodes = self.fit_node(nodes,labels,[],p,im_y,(level+ 1 == self.max_depth),X,N)
			level += 1
		return


	def measure_node(self,node,is_predict,p_list,X,i):

		while True:
			if type(self.tree[node]) == Counter:
				p_list.append(list(self.tree[node].keys())[np.argmax(self.tree[node].values())]
							  if is_predict else
							  {key: self.tree[node][key] / float(np.sum(list(self.tree[node].values()))) for
							   key in self.c_s})
				break
			else:
				node = node * 2 + (1 if X[self.tree[node][0]][i] < self.tree[node][1] else 2)
		return p_list

	def measure_pro(self, X,is_predict):

		p_list = []
		for i in range(len(X)):
			p_list = self.measure_node(0,is_predict,p_list,X,i)
		return p_list

	def predict(self, X):

		return self.measure_pro(X,True)

	def predict_proba(self, X):

		return pd.DataFrame(self.measure_pro(X,False), columns = self.c_s)



