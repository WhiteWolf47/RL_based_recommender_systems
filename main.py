import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF,LatentDirichletAllocation
import seaborn as sns
from tqdm import tqdm as tqdm_base
from gensim.models import HdpModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import umap
from data_cleaning import Data_Cleaning

# Any results you write to the current directory are saved as output.
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

movie_df = pd.read_csv('wiki_movie_plots_deduped.csv')

def group_genre(Genre_improved):
    '''
    After cleaning the Genre we have grouped similar set of genres together. For Example: action|comedy and 
    comedy|action were considered two different set of genres previously but in this function we have rectified it.
    We have also restricted our genre categories to some selected categories as mentioned in the list "list_genre".
    
    '''
    movie_df['Genre_grouped'] = movie_df['Genre_improved']
    list_genre = ['action','adult','animation','children','comedy','drama','fantasy','romance','supernatural',
                 'biography','history','thriller','science','mystery','series','artistic']
    for i in range(len(movie_df['Genre_improved'])):
        genre = movie_df['Genre_improved'][i]
        k = genre.split("|")
        k = set(k)
        k = sorted(k)
        k = [u for u in k if u in list_genre]
        k = [x for x in k if x]
        final = "|".join(k)
        movie_df['Genre_grouped'][i] = final
    movie_df['Genre_grouped'] = movie_df['Genre_grouped'].replace('','Default')
    return movie_df['Genre_grouped']

def pre_Process_data(documents):
    '''
    For preprocessing we have regularized, transformed each upper case into lower case, tokenized,
    Normalized and remove stopwords. For normalization, we have used PorterStemmer. Porter stemmer transforms 
    a sentence from this "love loving loved" to this "love love love"
    
    '''
    STOPWORDS = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    Tokenized_Doc=[]
    print("Pre-Processing the Data.........\n")
    for data in tqdm(documents):
        review = re.sub('[^a-zA-Z]', ' ', data)
        gen_docs = [w.lower() for w in word_tokenize(review)] 
        tokens = [stemmer.stem(token) for token in gen_docs if not token in STOPWORDS]
        final_=' '.join(tokens)
        Tokenized_Doc.append(final_)
    return Tokenized_Doc

def Vectorization(processed_data):
    '''
    Vectorization is an important step in Natural Language Processing. We have
    Used Tf_Idf vectorization in this script. The n_gram range for vectorization 
    lies between 2 and 3, that means minimum and maximum number of words in 
    the sequence that would be vectorized is two and three respectively. There
    are other different types of vectorization algorithms also, which could be added to this 
    function as required.
    
    '''
    vectorizer = TfidfVectorizer(stop_words='english', 
                                    max_features= 20000,#200000, # keep top 200000 terms 
                                    min_df = 1, ngram_range=(1,1), #(2,3),
                                    smooth_idf=True)
    X = vectorizer.fit_transform(processed_data)
    print("\n Shape of the document-term matrix")
    print(X.shape) # check shape of the document-term matrix
    return X, vectorizer

def topic_modeling(model,X):
    '''
    We have used three types of decomposition algorithm for unsupervised learning, anyone could 
    be selected with the help of the "model" parameter. Three of them are TruncatedSVD ,Latent
    Dirichlet Allocation and Matrix Factorization. This function is useful for comparing
    different model performances, by switching between different algorithms with the help of 
    the "model" parameter and also more algorithms could be easily added to this function.
    
    '''
    components = 16
    if model=='svd':
        print("\nTrying out Truncated SVD......")
        model_ = TruncatedSVD(n_components=components, algorithm='randomized', n_iter=1000, random_state=42)
        model_.fit(X)
    if model=='MF':
        print("\nTrying out Matrix Factorization......")
        model_ = NMF(n_components=components, random_state=1,solver='mu',
                      beta_loss='kullback-leibler', max_iter=1000, alpha=.1,
                      l1_ratio=.5).fit(X)
        model_.fit(X)
    if model=='LDA':
        print("\nTrying out Latent Dirichlet Allocation......")
        Tokenized_Doc=[doc.split() for doc in processed_data]
        dictionary = Dictionary(Tokenized_Doc)
        corpus = [dictionary.doc2bow(tokens) for tokens in Tokenized_Doc]
        model_ = LdaModel(corpus, num_topics=components, id2word = dictionary)
        '''model_ = LatentDirichletAllocation(n_components=components,max_iter=40,n_jobs=-1,
                                           random_state=42,verbose=0,learning_decay=0.3,
                                           learning_offset=30.
                                          )'''
        model_.fit(X)

    if model=="HDP":
        print("\nTrying out Hierarchical Dirichlet Process......")
        Tokenized_Doc=[doc.split() for doc in processed_data]
        dictionary = Dictionary(Tokenized_Doc)
        corpus = [dictionary.doc2bow(tokens) for tokens in Tokenized_Doc]
        model_ = HdpModel(corpus, id2word = dictionary)
        model_.fit(X)

    return model_

def Get_MostImportant_words(model, vectorizer):
    '''
    This function is used to evaluate top twenty most important words under each category.
    '''
    terms = vectorizer.get_feature_names()

    for i, comp in enumerate(model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:30]
        print("Category "+str(i)+": ")
        for t in sorted_terms:
            print(t[0],end =", ")
        print("\n")

def Visualize_clusters(model_, title):
    '''
    This function is used to visualize the clusters generated by our 
    model through unsupervised learning. We have used UMAP for better 
    visualization of clusters.
    
    '''
    X_topics = model_.fit_transform(X)
    embedding = umap.UMAP(n_neighbors=10,random_state=42).fit_transform(X_topics)#20

    plt.figure(figsize=(20,20))
    plt.title(title,fontsize=16)
    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = movie_df['Genre_grouped'],cmap='Spectral', alpha=1.0,
    s = 10, # size
    )
    plt.show()

movie_df['Genre_improved'] = Data_Cleaning(movie_df['Genre'])
movie_df['Genre_grouped'] = group_genre(movie_df['Genre_improved'])
movie_df = movie_df[movie_df['Genre_grouped']!='Default']# Defalut categories are removed
processed_data = pre_Process_data(movie_df['Plot'])

unique, counts = np.unique(movie_df['Genre_grouped'], return_counts=True)
for x,y in zip(unique,counts):
    print(x+" -> "+str(y))

movie_df['Genre_grouped'] = movie_df['Genre_grouped'].astype("category").cat.codes

X, vectorizer = Vectorization(processed_data)