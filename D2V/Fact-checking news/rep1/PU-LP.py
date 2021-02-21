import numpy as np
import pandas as pd
import networkx as nx
import sys


class PULP():
	def __init__(self, exp_metadata):
		graphml_input_file = sys.argv[2] 
		matrix_w_input_name = sys.argv[3] 

		self.output_file = sys.argv[4]

		self.train_folds = exp_metadata.fold.split(',')
		self.m = int(exp_metadata.m)
		self.lmbda = float(exp_metadata.l)
		self.dataset_file = exp_metadata.dataset

		self.G = nx.read_graphml(graphml_input_file)
		self.W = pd.read_csv(matrix_w_input_name, index_col=0)

		self.dataset = None
		self.train = None
		self.test = None
		self.labels = None


	def begin(self):

		print('Lendo arquivo tsv...')
		
		self.dataset = pd.read_csv(self.dataset_file, sep='\t', index_col=0, header=None)

		self.train, self.test, self.labels = self.train_teste()

		self.calc_pulp()


	def calc_pulp(self):
		#P = conjunto de exemplos positivos rotulados
		P = list(self.train.index.values)
		#U = conjunto não rotulado
		U = list(self.test.index.values)

		#RP = Reliable positie e RN = reliable negative
		RP = []
		Pcopy = P[:]
		Ucopy = U[:]

		print('Calculando ranks', flush=True)
		top = int((self.lmbda / self.m)*len(P))

		for self.k in range(0, self.m):
			#rank armazena os top exemplos mais similares ao conjunto P para adicioná-los a RP
			rank = np.zeros(top)
			rank_names = ['' for temp in range(top)]
			min_value = 0
			min_value_id = 0
			for vi in Ucopy:
				Svi = 0
				for vj in Pcopy:
					Svi += self.W.loc[vi][vj]
				Svi /= len(Pcopy)
				if (Svi > min_value):
					rank_names[min_value_id] = vi
					rank[min_value_id] = Svi
					min_value_id = np.argmin(rank)
					min_value = rank[min_value_id]
			for i in rank_names:
				Pcopy.append(i)
				Ucopy.remove(i)
				RP.append(i)
				U.remove(i)

		bottom = len(RP+P)
		#rank armazena os bottom exemplos menos similares ao conjunto P+RP para adicioná-los a RN
		rank = [10000 for temp in range(bottom)]
		RN = ['' for temp in range(bottom)]
		max_value = 10000
		max_value_id = 0
		for vi in U:
			Svi = 0
			for vj in (P + RP):
				Svi += self.W.loc[vi][vj]
			Svi /= len(P + RP)
			if (Svi < max_value):
				RN[max_value_id] = vi
				rank[max_value_id] = Svi
				max_value_id = np.argmax(rank)
				max_value = rank[max_value_id]

		#Salvar um conjunto desse (RP e RN) para cada fold, para cada conjunto de parâmetros
		print('Salvando conjuntos P, RP e RN')

		self.save_file(RP+P, RN)

		

	def save_file(self, RP, RN):
		f = open(self.output_file, 'w')
		for i in RP: 
			f.write(i + ':news\t1,0\n')

		for i in RN:
			f.write(i + ':news\t0,1\n')
		f.close()



	def train_teste(self):
		train = []
		for i in self.train_folds:
			file = 'folds/fold'+str(i)
			f = open(file, 'r')
			for row in f:
				index  = row.replace('\n','')
				train.append(index)
			f.close()

		print("Calculando treino e teste...")
		test = self.dataset.drop(train)
		train = self.dataset.loc[train]

		labels = []
		for index in test.index:
			label = self.dataset.loc[index][2]
			if label == 1:
				labels.append(1)
			else:
				labels.append(-1)
		return train, test, labels


#Parâmetros de entrada
exp_id = sys.argv[1]
graphml_input_file = sys.argv[2] 

metadata = graphml_input_file.split('/')[:3]
metadata = '/'.join(metadata)
exp_metadata = pd.read_csv(metadata+'/params_pulp.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

pulp = PULP(exp_metadata)
pulp.begin()

