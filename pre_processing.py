import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import scipy.stats as stats
from nltk import FreqDist
from nltk.util import ngrams # function for making ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer 
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy import special, stats

NEGATION = ["not", "no", "nothing", "never", "none"]

# Trata palavras com apostrofo
def remove_not_alpha(words):
    
    only_alpha = [word.replace("'", "o") for word in words if word.isalpha() or (word == "n't")]
    
    return only_alpha
   
# Remove palavras irrelevantes
def remove_stop_words(words):
    
    stop_words = stopwords.words('english')
        
    without_sw = [word for word in words if word not in stop_words or word in NEGATION]
    
    return without_sw

# Executa o algoritmo de stemming, removendo o sufixo das palavras
def stemming(words):
    
    stemmer = PorterStemmer() 
   
    stemming_words = [ stemmer.stem(word) for word in words]
    
    return stemming_words

# funcao do nltk para 'tagging' de palavras individualmente
def pos_tag(words, noun):
        
    pt_words = []
       
    i = 0
    
    adj = False
    
    for token, pos in nltk.pos_tag(words):
        
        #if flag is true, get nouns
        if noun and pos[0] == 'N':
            pt_words.append(token)
        
        #get only verbs, adverbs and adjectives
        if pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R': 
            pt_words.append(token)
        
    return pt_words
    
# funcoes de tratamento    
def clear_text(words, noun):
        
    words = remove_not_alpha(words)

    words = remove_stop_words(words)

    words = stemming(pos_tag(words, noun))
        
    return words
         
# trata palavras de negacao, as adicionando em bigramas com a proxima palavra
def handle_negation(words):
           
    with_negation = []
    
    for i in range(len(words)):
        
        if words[i] in NEGATION and i+1 < len(words):
            with_negation.append((words[i], words[i+1]))
            i += 1
        else:
            with_negation.append(words[i])
            
    return with_negation

# le uma base de dados do disco, trata e retorna uma matriz esparsa com as palavras, classes e o vocabulario
def bow(category, hNeg=True, noun=False):
           
    category += '/'

    # Carrega as reviews negativas e positivas
    positive = open('sorted_data_acl/' + category  + 'positive.review', 'r')
    negative = open('sorted_data_acl/' + category  + 'negative.review', 'r')

    # Utiliza o BeautifulSoup para ler do xml
    positive_reviews = (BeautifulSoup(positive, 'lxml'))
    negative_reviews = (BeautifulSoup(negative, 'lxml'))

    # Guarda apenas as reviews 
    positive_reviews = positive_reviews.find_all(['review'])
    negative_reviews = negative_reviews.find_all(['review'])

    n_pos_reviews = len(positive_reviews)
    n_neg_reviews = len(negative_reviews)

    # Inicializa a bag of words, vocabulario e os bigramas
    bags = []
    vocabulary = []    
    bigrams = []

    # Trata as reviews positivas
    for review in positive_reviews:

        # Guarda apenas o titulo e o texto de cada review
        review_text = (review.find('title').string + review.find('review_text').string).lower()

        # Tokenize
        review_text = nltk.word_tokenize(review_text)

        # Chama a funcao que faz o tratamento dos dados
        review_text = clear_text(review_text, noun)
        
        # Guarda as palavras no vocabulario
        vocabulary.extend(review_text)
    
        bag = {}
        
        # Conta quantas ocorrencias de cada palavra
        for word in review_text:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1
        
        # Guarda as ocorrencias de cada palavra
        bags.append(bag)
                 
          
    for review in negative_reviews:

        # Guarda apenas o titulo e o texto de cada review
        review_text = (review.find('title').string + review.find('review_text').string).lower()

        # Tokenize
        review_text = nltk.word_tokenize(review_text)

        # Chama a funcao que faz o tratamento dos dados
        review_text = clear_text(review_text, noun)
        
        # Chama a funcao para juntar negacoes em bigramas
        if hNeg:
            review_text = handle_negation(review_text)
             
        # Guarda as palavras no vocabulario
        vocabulary.extend(review_text)
        
        bag = {}

        # Conta quantas ocorrencias de cada palavra
        for word in review_text:
            if word in bag:
                bag[word] += 1
            else:
                bag[word] = 1
        
        # Guarda as ocorrencias de cada palavra
        bags.append(bag)
        
    n_reviews = n_pos_reviews + n_neg_reviews
    
    # sort and get unique words
    vocabulary = list(set(vocabulary))
    
    # generates matrix where m[i][j] is the number of times the word j appears in document i
    matrix = np.zeros((n_reviews, len(vocabulary)), dtype="int")
      
    # Organiza as 'bags of words' em features e salva na matriz com as colunas correspondentes
    for i in range(n_reviews):
        for key in bags[i]:
            index = vocabulary.index(key)
            matrix[i][index] = bags[i][key]
    
    # make target array
    target = np.zeros((n_pos_reviews + n_neg_reviews), dtype="int")
    target[:n_pos_reviews] = 1
     
    #transform matrix in a sparse matrix    
    sMatrix = csr_matrix(matrix) 
        
    return sMatrix, target, vocabulary


def fp(X):
    return np.where(X > 0, 1, 0)


def tf_idf(X):
    
    m, n = X.shape 
    
    return X * np.log(m/np.count_nonzero(X, axis=0))
  
def chisquare(f_obs, f_exp):
    
    """
    Fast replacement for scipy.stats.chisquare.
    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    
    """
    
    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
        
    chisq = chisq.sum(axis=0)
    
    pvalue = special.chdtrc(k - 1, chisq)
    
    return pvalue

# Chi-quadrado para reducao de dimensionalidade
# seleciona apenas as features mais relevantes, baseado num parametro alpha
def chi2(X, y, vocabulary, alpha=0.05):
    
    """
    Based on sklearn.feature_selection.chi2
    
    """
    
    Y = np.zeros((len(y), 2), dtype=int) 
    for i in range(len(Y)):
        Y[i][y[i]] = 1
 
    observed = Y.T * X  # n_classes * n_features

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    
    expected = np.dot(class_prob.T, feature_count)

    pvalue = chisquare(observed, expected)
    
    new_vocabulary = []
    
    index = [i for i in range(len(pvalue)) if pvalue[i] < alpha]
    
    new_vocabulary = [vocabulary[i] for i in index]
    
    Xnew = X[:, index]
            
    return Xnew, new_vocabulary, index