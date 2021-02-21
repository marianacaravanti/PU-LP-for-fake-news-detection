import sys, json, pickle
import numpy as np
from hashlib import md5
from util.functions import *

#Bag-of-words parameters
representation_model = "Bag-of-words"
arg_tsv_file = "dataset/Fact_checked_news.tsv"
arg_stopwords = "dataset/stopwords.txt"
arg_language = "portuguese"
folds = 10
arg_options = ['2']
arg_ngram_range = ['1,1'] #['1,1', '1,2']
arg_min_df = ['2'] #unigram, bigram
arg_norm = ['l2']
#Min tf-idf value to remove 
#terms of knn networks
arg_min_weigth = 0.08

#k-nn matrix parameters
arg_k = ['5', '6', '7']
#W matrix parameters
arg_a = ['0.005', '0.01', '0.02']

#Parâmetros do PU-LP
arg_l = [0.6, 0.8]
arg_m = [2]

#GNM and LPHN parameters
arg_mi = [0.5, 1] 
arg_max_iter = 1000
agr_limiar_conv = 0.00005
weight_relations = {"news_term": 1, "news_news": 1}

output_file_dict = 'dataset/'
dir_base = 'scripts/'
dir_bow_op = 'bagofwords/op={}/{}/'
dir_pulp_output = dir_bow_op+'pulp_{}0%/'
dir_LPHN = dir_bow_op+'label_propagation_LPHN/'
dir_GNM = dir_bow_op+'label_propagation_GNM/'
dir_GNM_mi = dir_GNM + 'mi={}/'
output_heterg_gnm = dir_GNM_mi + '{}.model'
output_heterg_lphn = dir_LPHN + '{}.model'

dir_graph = dir_bow_op + 'representation_input/{}/'
dir_matrix = dir_graph + '{}/'
relations = dir_graph + 'doc_term.relations' 

graph = dir_graph + 'knn_fakenews.graphml'
w_matrix = dir_matrix + 'w_matrix.csv'
dir_fila_bow = dir_base + 'fila_bow/'
dir_fila_pu_lp = dir_base + 'fila_pulp/'
dir_params_pulp = dir_bow_op + 'params_pulp.metadata'
dir_params_prop = dir_bow_op + 'params_labelprop.metadata'

create_path(dir_base)
create_path(dir_fila_bow)
create_path(dir_fila_pu_lp)
create_path('logs')
create_path('folds')

nucleos_bow_adj, multiple = sys.argv[1:]
nucleos_bow_adj = int(nucleos_bow_adj)
multiple = int(multiple) 
nucleos_pulp = multiple

params = open('params.metadata', 'w')
params.write('id\tdataset\trepresentation_model\tstopwords\tlanguage\toption\tngram_range\tmin_df\tnorm\targ_min_weigth\n')

dataset = {'1': {}, '-1': {}}
f = open(arg_tsv_file, 'r')

for line in f:
	index, text, label = line.split('\t')
	label = label.replace('\n', '')
	dataset[label][index] = text
f.close()


count = 0
for index in dataset['1']:
	fold = count%folds
	f = open('folds/fold'+str(fold), 'a')
	f.write(index + "\n")
	f.close()
	count+=1

dict_file = open(output_file_dict+'dict_dataset.pkl', 'wb')
pickle.dump(dataset, dict_file)
dict_file.close()


count = 0
count_pulp = 0

list_fila_dir = []
list_fila_pulp = []
for op in arg_options:
	for ngr in arg_ngram_range:
		for mdf in arg_min_df:
			for nrm in arg_norm:
				#for rds in arg_random_state:
					#for ts in arg_test_size:
						#for mmi in arg_min_mi:
							
				dir = count % nucleos_bow_adj
				fbow = dir_fila_bow + 'fila_{}.sh'.format(dir)	
				
				if not dir in list_fila_dir: 
					list_fila_dir.append(dir)
					fb = open(dir_fila_bow+'fila_bow.sh', 'a')
					fb.write('nohup {} > logs/fila_bow_{}.log &\n'.format(fbow, dir))	
					fb.close()

					
				
				exp_id = md5('{} {} {} {} {} {} {} {} {}'.format(arg_tsv_file, representation_model, arg_stopwords, arg_language,
					op, ngr, mdf, nrm, arg_min_weigth).encode()).hexdigest()
				
				params.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(exp_id, arg_tsv_file, representation_model, 
					arg_stopwords, arg_language, op, ngr, mdf, nrm, arg_min_weigth))
			
				f = open(fbow, 'a')
				f.write("python3 bagofwords.py {} {} > logs/{}-bow.log\n".format(exp_id, dir_bow_op, exp_id))
				f.close()							
				
				f = open(fbow, 'a')
				f.write("python3 adjacency_matrix.py {} {} {} {} {} > logs/{}-adj.log\n".format(exp_id, dir_bow_op, ','.join(arg_k),','.join(arg_a), arg_min_weigth, exp_id))
				f.close()
				
				count+=1
				

				create_path(dir_GNM.format(op,exp_id))
				create_path(dir_LPHN.format(op,exp_id))
				
				params_pulp = open(dir_params_pulp.format(op, exp_id), 'w')
				params_pulp.write('id\tdataset\trepresetation\tfold\tk\ta\tm\tl\n')

				params_prop = open(dir_params_prop.format(op, exp_id), 'w')
				params_prop.write('id\tpulp_id\tfold\tmi\tmax_iter\tlimiar_conv\tweight_relations\n')

				count_multiple = 0
				for p in range(3):

					create_path(dir_pulp_output.format(op,exp_id,p+1))

					for k in arg_k:
						input_graph = graph.format(op, exp_id, k)
						for a in arg_a:
							path_dir_matrix = dir_matrix.format(op,exp_id,k,a)	
							input_matrix_w = w_matrix.format(op, exp_id, k, a)
							create_path(path_dir_matrix)
							
							fold = ''										
										
							for i in range(10):

								for m in arg_m:

									for l in arg_l:													

										if p == 0: fold = '{}'.format(i%10)
			
										if p == 1: fold = '{},{}'.format(i%10, (i+1)%10)
											
										if p == 2: fold = '{},{},{}'.format(i%10, (i+1)%10, (i+2)%10)
								
										pulp_id = md5('{} {} {} {} {} {} {}'.format(arg_tsv_file, representation_model, fold, k, a, m, l).encode()).hexdigest()
										params_pulp.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(pulp_id, arg_tsv_file, representation_model, fold, k, a, m, l))
										#, input_graph, input_matrix_w
										dir_pulp = count_pulp % nucleos_bow_adj
										dir_pulp_multiple = count_multiple % multiple
										pu_lp_output = dir_pulp_output+'{}.labels'
										pu_lp_output = pu_lp_output.format(op, exp_id, p+1, pulp_id)
										file_name = dir_fila_pu_lp+'fila_{}_{}.sh'.format(dir_pulp, dir_pulp_multiple)
										
										f = open(file_name, 'a')
										f.write("python3 PU-LP.py {} {} {} {}  > logs/{}_{}-pulp.log\n". format(pulp_id, input_graph, input_matrix_w, pu_lp_output, exp_id, pulp_id))
																							
										

										for mi in arg_mi:
											create_path(dir_GNM_mi.format(op, exp_id, mi))
											
											prop_id = md5('{} {} {} {} {} {}'.format(pulp_id, fold, mi, arg_max_iter, agr_limiar_conv, weight_relations).encode()).hexdigest()
				
											params_prop.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(prop_id, pulp_id, fold, mi, arg_max_iter, agr_limiar_conv, weight_relations))													

											f.write("java -Xmx5G -cp ClassificadorTrasdutivo.jar algoritmos.RegularizarFakeNews {} {} {} {} {} {} {} > logs/{}_{}-gnm.log\n".format('gn', prop_id, pulp_id, relations.format(op, exp_id, k), 
												dir_params_prop.format(op, exp_id), output_heterg_gnm.format(op, exp_id, mi, pulp_id), dir_pulp_output.format(op, exp_id, p+1), exp_id, pulp_id)) 
											
											
											f.write("python3 results_GNM.py {} {} {} {} {} > logs/{}_{}-resuts-gnm.log\n".format(exp_id, pulp_id, prop_id, output_heterg_gnm.format(op, exp_id, mi, pulp_id),  dir_bow_op.format(op, exp_id), exp_id, pulp_id))

										
										
										f.write("java -Xmx5G -cp ClassificadorTrasdutivo.jar algoritmos.RegularizarFakeNews {} {} {} {} {} {} {} > logs/{}_{}-gnm.log\n".format('lp', prop_id, pulp_id, relations.format(op, exp_id, k), 
											dir_params_prop.format(op, exp_id), output_heterg_lphn.format(op, exp_id, pulp_id), dir_pulp_output.format(op, exp_id, p+1), exp_id, pulp_id)) 

										
										f.write("python3 results_LPHN.py {} {} {} {} {} > logs/{}_{}-results-lp.log\n".format(exp_id, pulp_id, prop_id, output_heterg_lphn.format(op, exp_id, pulp_id), dir_bow_op.format(op, exp_id), exp_id, pulp_id))

										f.write('\necho "---------- proximo ----------"\n\n')
										
										f.close()
										count_multiple +=1
				
				count_pulp += 1	
											

count_pulp = 0
list_fila_pulp = []

for op in arg_options:
	for ngr in arg_ngram_range:
		for mdf in arg_min_df:
			for nrm in arg_norm:
				count_multiple = 0
				for p in range(3):
					for k in arg_k:
						for a in arg_a:
							for i in range(10):
								for m in arg_m:
									for l in arg_l:	
										dir_pulp = count_pulp % nucleos_bow_adj
										dir_pulp_multiple = count_multiple % multiple
										string = str(dir_pulp)+'_'+str(dir_pulp_multiple)
										if not string in list_fila_pulp: 
											list_fila_pulp.append(string)
											dir_pulp = count_pulp % nucleos_bow_adj
											dir_pulp_multiple = count_multiple % multiple
											output_fila_bow = dir_fila_bow + 'fila_{}.sh'.format(dir_pulp)	
											fb = open(output_fila_bow, 'a')
											file_name = dir_fila_pu_lp+'fila_{}_{}.sh'.format(dir_pulp, dir_pulp_multiple)
											fb.write('nohup {} > logs/fila_pulp_{}_{}.log &\n'.format(file_name, dir_pulp, dir_pulp_multiple))	
											fb.close()
										
											count_multiple += 1
				count_pulp += 1	
						

							
params.close()							
params_pulp.close()		
params_prop.close()					
							

					
							
