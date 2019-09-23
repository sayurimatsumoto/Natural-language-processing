import numpy as np
import pandas as pd

# Calcula e retorna a matriz de confusão dada uma base de dados
def get_confusionMatrix(Y_test, Y_pred, classes):
    
    cm = np.zeros( [len(classes),len(classes)], dtype=int )

    n_classes = len(classes)
            
    for exp, pred in zip(Y_test, Y_pred):
        cm[exp][pred] += 1
                
    return cm

# Calcula e retorna um relatorio com as principais medidas
def relatorioDesempenho(Yval, Ypred, classes, imprimeRelatorio=False):
    
    matriz_confusao = get_confusionMatrix(Yval, Ypred, classes)
        
    n_teste = sum(sum(matriz_confusao))

    nClasses = len( matriz_confusao ) #numero de classes

    # inicializa as medidas que deverao ser calculadas
    vp=np.zeros( nClasses ) # quantidade de verdadeiros positivos
    vn=np.zeros( nClasses ) # quantidade de verdadeiros negativos
    fp=np.zeros( nClasses ) # quantidade de falsos positivos
    fn=np.zeros( nClasses ) # quantidade de falsos negativos
    
    acuracia = 0.0 

    revocacao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    revocacao_macroAverage = 0.0
    revocacao_microAverage = 0.0

    precisao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    precisao_macroAverage = 0.0
    precisao_microAverage = 0.0

    fmedida = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
    fmedida_macroAverage = 0.0
    fmedida_microAverage = 0.0
    
    vp = np.diag(matriz_confusao)
    fp = np.sum(matriz_confusao, axis = 0) - vp
    fn = np.sum(matriz_confusao, axis = 1) - vp
    vn = np.sum(matriz_confusao) - (fp + fn + vp)

    
    acuracia = sum(vp + vn)/sum(vp + vn + fp + fn)
    revocacao = vp/(vp + fn)
    precisao = vp/(vp + fp)
    fmedida = 2 * ((precisao * revocacao)/(precisao + revocacao))
    
    revocacao_macroAverage = revocacao.mean()
    revocacao_microAverage = sum((vp + vn))/sum((vp + vn + fp + fn))
    
    precisao_macroAverage = precisao.mean()
    precisao_microAverage = sum(vp)/sum((vp + fp))
    
    fmedida_macroAverage = 2 * ( (revocacao_macroAverage * precisao_macroAverage)/(revocacao_macroAverage + precisao_macroAverage)) 
    fmedida_microAverage = 2 * ( (revocacao_microAverage * precisao_microAverage)/(revocacao_microAverage + precisao_microAverage)) 
    
    
  ##################################################################################    

# imprimindo os resultados para cada classe
    if imprimeRelatorio:
        print('\n\tRevocacao   Precisao   F-medida   Classe')
        for i in range(0,nClasses):
            print('\t%1.3f       %1.3f      %1.3f      %s' % (revocacao[i], precisao[i], fmedida[i],classes[i] ) )

        print('\t------------------------------------------------');

        #imprime as médias
        print('\t%1.3f       %1.3f      %1.3f      Média macro' % (revocacao_macroAverage, precisao_macroAverage, fmedida_macroAverage) )
        print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (revocacao_microAverage, precisao_microAverage, fmedida_microAverage) )

        print('\tAcuracia: %1.3f' %acuracia)

    # guarda os resultados em uma estrutura tipo dicionario
    resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao':precisao, 'fmedida':fmedida}
    resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
    resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
    resultados.update({'confusionMatrix': matriz_confusao})

    return resultados 

# Faz o holdout estratificado, dado uma base e a porcentagem de treino
# Retorna os indices de treino e teste, mantendo as proporcoes originais de cada classe
def stratified_holdOut(target, pTrain):
    
    train_index = []
    test_index = []
    
    classes = np.unique(target)
    i = []
    for c in classes:
        i, = np.where((target == c))
        p = round(pTrain*len(i))
        train_index.extend(i[:p])
        test_index.extend(i[p:])    
    
    train_index.sort()
    test_index.sort()    
    
    return train_index, test_index



