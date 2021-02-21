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

#from liwc.liwc import Liwc
#liwc_pt = Liwc('dictionaries/LIWC2007_Portugues_win.dic')
#liwc_en = Liwc('dictionaries/LIWC2007_English.dic')
