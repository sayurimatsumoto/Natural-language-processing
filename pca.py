# -*- coding: utf-8 -*-

import numpy as np # biblioteca usada para trabalhar com vetores e matrizes
import pandas as pd # biblioteca usada para trabalhar com dataframes
from scipy.sparse.linalg import svds, eigs
from scipy.linalg import svd
from scipy.sparse import csr_matrix

def pca(X):
    """
    Executa a analise de componentes principais na base de dados X

    Retorna os autovetores U e os autovalores em S.
    """
    
    m, n = X.shape # m = qtde de objetos e n = qtde de atributos
    
    # incializa as variaves de saida
    U = np.zeros( [n,n] )
    S = np.zeros( n )    
    
    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: você precisa calcular a matriz de covariancia de X e,
    #                posteriormente, usar a função svd para calcular os 
    #                autovetores e autovalores da matriz encontrada.
    #
    #                A funcao devera retornar os autovalores e autovetores
    #                calculados pela funcao np.linalg.svd  

    
    matrix = (1/m) * np.dot(X.T, X)
    
    U, S, v = svd(matrix)    
    
    ########################################################################## 

    return U, S 


def projetarDados(X, U, K): 
    """
    Computa a representacao reduzida pela projecao usando os K
    primeiros autovetores
    
    Calcula a projecao dos dados X em um espaco
    de dimensao reduzida gerado pelas primeiras K colunas de
    U. A funcao retorna os exemplos projetados em Z.
    """

    # incializa a variavel de saida
    Z = np.zeros( [X.shape[0],K] )   

    ########################## COMPLETE O CÓDIGO AQUI  ########################
    #  Instrucoes: você precisa computar a projecao dos dados usando apenas os 
    #                primeiros K autovetores em U. 
    # 
    
    Z = csr_matrix(X * U[:, :K])

    return Z

# U, S = pca.pca(Xtrain.toarray())

# for k in range(100, Xtrain.shape[1]):
#     if S[:k][:k].sum()/S.sum() >= 0.99:
#         K = k
#         break
        

# Xtrain = pca.projetarDados(Xtrain, U, K)
# Xval = pca.projetarDados(Xval, U, K)
