from util.bibliotecas import *
from util.functions import *


def convert(vector, type):
    vector = [type(i) for i in vector]
    return vector

def domain_stopwords(stopwords=None):
    #get domain stopwords
    domain_stopwords = ""
    if stopwords==None:
        return domain_stopwords
    f = open(stopwords, "r")
    for line in f:
        domain_stopwords +=  line.strip() + " "
    f.close()
    return word_tokenize(domain_stopwords)

class BoW():
    def __init__(self, exp_id, exp_metadata):
        
        self.exp_id = exp_id
        self.tsv_file = exp_metadata.dataset  #dataset.tsv
        self.stopwords = exp_metadata.stopwords #stopwords.txt
        self.language = exp_metadata.language  #portuguese or english
        self.option = exp_metadata.option #1 (remove stopwords) or 2 (stopwords and stemming)
        self.ngram_range = exp_metadata.ngram_range #1,1; 1,2; 2,2 (unigrams or unigrams and bigrams)
        self.min_df = exp_metadata.min_df #0.001
        self.norm = exp_metadata.norm   #l1, l2
        
        self.output_file = sys.argv[2]

        self.option = int(self.option)
        self.min_df = int(self.min_df)
        
        self.domain_stopwords = domain_stopwords(self.stopwords)
        self.documents = None
        self.indexes = None
        self.labels = None
        self.tfidf_vect_df = None
        self.begin()
    


    def begin(self):
        
        self.preprocess() 
        
        
        x,y = self.ngram_range.split(',')
        x,y = int(x), int(y)

        
        matrix =  TfidfVectorizer(ngram_range=(x,y), min_df=self.min_df, norm=self.norm)
        X = matrix.fit_transform(self.documents)
        self.tfidf_vect_df = pd.DataFrame(X.todense(), columns=matrix.get_feature_names(), index=self.indexes)
        
               
        file_path = self.output_file.format(self.option, self.exp_id)
        print('tfidf matrix dimensions: ', self.tfidf_vect_df.shape)
        print(file_path)
        create_path(file_path)
        self.tfidf_vect_df.to_csv(file_path+'tf_idf_vect.csv')
            

    def preprocess(self):
        
        self.labels = []
        self.documents = []
        self.indexes = []

        print('Reading news dataset...')
        f = open(self.tsv_file, 'r')
        
        print('Collecting dataset information...')
        for line in f:
            index, txt, label = line.split('\t')
            label = label.strip()
            #Option: 1: remove stopwords, 2: remove stopwords and stemming
            self.labels.append(label)
            text = self.remove_stopwords(text=txt)
            if (self.option is 2):
                text = self.stemming(text.strip())
            #Adicionando o texto ao corpus para treinamento
            self.documents.append(text.strip())
            self.indexes.append(index)
        
        file_path = self.output_file.split('/')[:2]
        file_path = "/".join(file_path).format(self.option, self.exp_id)
        create_path(file_path)
        np.save(file_path+'/indexes',self.indexes)
        np.save(file_path+'/documents',self.documents)
        np.save(file_path+'/labels',self.labels)

        f.close()
        

    def remove_stopwords(self,text):
        stop_words = nltk.corpus.stopwords.words(self.language)
        s = str(text).lower() # lower case
        table = str.maketrans({key: None for key in string.punctuation})
        s = s.translate(table) # remove punctuation
        tokens = word_tokenize(s) #get tokens
        v = []
        for i in tokens:
            if not (i in stop_words or i in self.domain_stopwords or i.isdigit() or len(i)<= 1): # remove stopwords
                v.append(i)
        s = ""
        for token in v:
            s += token + " "
        return s

    def stemming(self, text):
        stemmer = PorterStemmer() # stemming for English
        if self.language=='portuguese':
            stemmer = nltk.stem.RSLPStemmer() # stemming for portuguese
        tokens = word_tokenize(text) 
        sentence_stem = ''
        doc_text_stems = [stemmer.stem(i) for i in tokens]
        for stem in doc_text_stems:
            sentence_stem += stem+" "
        return sentence_stem.strip()

                                

exp_id = sys.argv[1]
exp_metadata = pd.read_csv('params.metadata', sep='\t', index_col=0)
exp_metadata = exp_metadata.loc[exp_id]

bow = BoW(exp_id, exp_metadata)




