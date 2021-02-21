from util.bibliotecas import *
from util.functions import *

class MatricesCalc():
	def __init__(self, exp_id, exp_metadata):

		option = exp_metadata.option
		input_path = sys.argv[2].format(option,exp_id)
		input_files = input_path.split('/')[:2]
		input_files = "/".join(input_files)
		self.indexes_file = input_files + '/indexes.npy'
		self.bow_file = input_path+'tf_idf_vect.csv'
		self.adj_matrix_file = input_path+'representation_input/adj_matrix.csv'
		self.graphml_output_file = input_path+'representation_input/{}/knn_fakenews.graphml'
		self.doc_term_relations_output = input_path+'representation_input/{}/doc_term.relations' 	
		self.output_matriz_w = input_path+'representation_input/{}/{}/w_matrix.csv'
				
		arg_k = sys.argv[3].split(',')
		arg_a = sys.argv[4].split(',')
		
		self.arg_k = [int(x) for x in arg_k]
		self.arg_a = [float(x) for x in arg_a]
		self.arg_min_weigth = float(sys.argv[5])
		self.indexes = np.load(self.indexes_file)

		self.tfidf_vect_df = pd.read_csv(self.bow_file, sep=',', index_col=0, header=0)

		self.adjacence_matrix = self.calc_MatrizAdj()
		self.ident_matrix = self.calc_Indent()
		self.A = None
		self.G = None
		self.calc_knn_w_matrices()		

		

	def calc_Indent(self):
		print("Calculando matriz identidade...", flush=True)
		ident_matrix = np.identity(len(self.adjacence_matrix), dtype = float)
		ident_matrix = pd.DataFrame(ident_matrix, columns=self.indexes, index=self.indexes)
		return ident_matrix

	
	def calc_MatrizAdj(self):
		adj_matrix = []
		for id in self.indexes:
			adj_matrix.append(self.tfidf_vect_df.loc[id].tolist())

		adj_matrix = pd.DataFrame(adj_matrix, index=self.indexes)

		time_init = time()
		Y = cdist(adj_matrix, adj_matrix, metric=cosine)
		adjacence_matrix = pd.DataFrame(Y, columns=self.indexes, index=self.indexes)
		adjacence_matrix.to_csv(path_or_buf=self.adj_matrix_file)
		return adjacence_matrix

	
	def calc_knn_w_matrices(self):
		print("Calculating k-nn matrices...", flush=True)
		for valor_k in self.arg_k:
			print(valor_k)

			print("Calculando {}-NN...".format(valor_k), flush=True)
			self.A, self.G = self.PU_LP_knn(valor_k)

			self.doc_term_net(valor_k)
			
			for alfa in self.arg_a:
				self.calc_W(valor_k, alfa)
	
	def calc_W(self, valor_k, alfa):
		matrix_w_output_name = self.output_matriz_w.format(valor_k, alfa)
		
		print("\tCalculando Índice de Katz aplha={}...".format(alfa), flush=True)
		#Calcula W = (I - alfa x A)^-1 - I
		#Medida de centralidade que considera número de vizinhos de um nó e o quão importante esses vizinhos são
		A = self.A.mul(alfa)
		I = self.ident_matrix.subtract(A)
		
		I = pd.DataFrame(np.linalg.pinv(I.values.astype(np.float32)), columns=self.indexes, index=self.indexes)
		W = I.subtract(self.ident_matrix)

		W.to_csv(path_or_buf=matrix_w_output_name)

		print('\t\tDesalocar Matriz W', flush=True)
		del W			

	def PU_LP_knn(self, k):
		
		total_columns = self.adjacence_matrix.shape[0]
		#A é uma matriz com base em k-NN, na qual Aij = 1 se j é um dos k vizinhos de i, e 0 c.c.
		A = pd.DataFrame(0, columns=self.adjacence_matrix.index, index=self.adjacence_matrix.index)
		#G é o grafo gerado a partir da matriz A, cujos vértices não estão rotulados
		G = nx.Graph()
		for index_i, row in self.adjacence_matrix.iterrows():
			#knn é um vetor de k posições, inicializadas com valor alto
			#o vetor knn armazena os k índices de vizinhos mais próximos do vértice i (menor distância de cosseno)
			knn = [1000 for temp in range(k)]
			#knn_names armazena os nomes correspondentes aos índices armazenados no vetor knn
			knn_names = ['' for temp in range(k)]
			max_value = 1000
			max_value_id = 0
			#para cada coluna correspondente a linha i (j vizinhos)
			for name_j, value in row.iteritems():
				#se i != j faça
				if(index_i != name_j):
					#se a distância é menor que o maior valor armazenado no vetor knn:
					if (value < max_value):
						#adiciono vértice j aos vizinhos de i
						knn_names[max_value_id] = name_j
						knn[max_value_id] = value
						max_value_id = np.argmax(knn)
						max_value = knn[max_value_id]
			for j in range(k):
				#Seleciona os 4 vizinhos mais próximos e os adiciono na matriz A
				vizinho = knn_names[j]
				A.loc[index_i][vizinho] = 1
				#Adiciona aresta no grafo entre i e seus 4 vizinhos mais próximos
				#print(index_i, '\t', vizinho)
				G.add_edge(index_i, vizinho, weight=1)
		
		print("\tSalva informações do Grafo")
		nx.write_graphml(G, self.graphml_output_file.format(k))
		return A, G


	def doc_term_net(self, valor_k):

		doc_term_relations_output = self.doc_term_relations_output.format(valor_k)
		fwrite = open(doc_term_relations_output, 'w')
		
		for x, y in self.G.edges:
			node_1 = x + ':news'
			node_2 = y + ':news'
			weight = 1
			edge = node_1 + "\t" + node_2 + "\t" + str(weight) + "\n"
			fwrite.write(edge)

		for term in self.tfidf_vect_df.columns:
			for i in range(len(self.indexes)):
				news = self.indexes[i]
				weight = self.tfidf_vect_df.loc[news,term]
				if weight > self.arg_min_weigth:
					node_1 = news + ':news'
					node_2 = term + ':term'      
					edge = node_1 + "\t" + node_2 + "\t" + str(weight) + "\n"
					fwrite.write(edge)

		fwrite.close()

#Parâmetros de entrada
exp_id = sys.argv[1]
exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

mtc = MatricesCalc(exp_id, exp_metadata)

