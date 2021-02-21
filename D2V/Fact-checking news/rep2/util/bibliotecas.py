import logging, math, gensim, json
import string, re, collections
import sys, pickle
import pandas as pd
import numpy as np
import networkx as nx
from time import time
from os import path
import datetime as dt
from abc import abstractmethod
import joblib

#Nltk
import nltk
from nltk import word_tokenize
from nltk import download
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.metrics import ConfusionMatrix, precision, recall, f_measure, accuracy
nltk.download('punkt')
download('stopwords') 
nltk.download('rslp')
stop_words = stopwords.words('portuguese')

#Sklearn
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, SelectPercentile
from sklearn.model_selection import train_test_split

#Gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#Scipy
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

class StructuredRepresentation():

    def __init__(self, doc_vectors=None, class_vectors=None, vocabulary=None): 
        self.text_vectors = doc_vectors
        self.class_vectors = class_vectors 
        self.vocabulary = vocabulary

  
    def save_arff(self, name, path, non_sparse_format = False):
        num_docs = self.text_vectors.shape[0]
        num_attrs = self.text_vectors.shape[1]
        with open(path, 'w') as arff: 
            #Writting the relation
            arff.write('@relation {}\n\n'.format(name))
            
            #Writting the attributes
            if self.vocabulary == None: 
                for attr in range(num_attrs): 
                    arff.write('@ATTRIBUTE dim{} NUMERIC\n'.format(attr + 1))
            else: 
                sorted_vocabulary = sorted(self.vocabulary.items(), key=lambda x: x[1])
                for attr in range(num_attrs): 
                    arff.write('@ATTRIBUTE {} NUMERIC\n'.format(sorted_vocabulary[attr][0]))
            
            #Writting the class names
            arff.write('@ATTRIBUTE att_class ' + '{"' + '","'.join(self.class_vectors.unique()) + '"}\n\n')


            #Writting the data
            arff.write('@data\n\n')


            if non_sparse_format == False: 
                for doc in range(num_docs):
                    vector = self.text_vectors[doc]
                    if type(vector) == scipy.sparse.csr.csr_matrix: 
                        vector = self.text_vectors[doc].toarray()[0]
                    str_vec = ''
                    for i in range(vector.shape[0]): 
                        str_vec += str(vector[i]) + ','
                    classe = self.class_vectors.iloc[doc]
                    arff.write(str_vec + '"' + classe + '"\n') 
            else: 
                for doc in range(num_docs):
                    vector = self.text_vectors[doc]
                    if type(vector) == scipy.sparse.csr.csr_matrix: 
                        vector = self.text_vectors[doc].toarray()[0]
                    str_vec = ''
                    for i in range(vector.shape[0]): 
                        if vector[i] > 0: 
                            str_vec += '{i} {},'.format(str(vector[i]))
                    classe = self.class_vectors.iloc[doc]
                    arff.write('{' + str_vec + str(num_attrs) + ' "' + classe + '"}\n') 

def load_representation(path): 
    return joblib.load(path)
