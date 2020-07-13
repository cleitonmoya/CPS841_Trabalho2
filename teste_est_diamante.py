# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
Porgrama de Pos-Graduação em Engenharia de Sistemas e Computação
CPS841 - Redes Neurais Sem Peso

Teste da WiSARD para predição de tendências de ações com a estratégia 
"diamante, conforme o artigo "Análise de Séries Temporais Financeiras 
Utilizando WiSARD"

@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import time
import datetime
import wisardpkg as wp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# **************************** FUNÇÕES AUXILIARES ****************************

# Formata o dataframe para o formato da biblioteca pandas_ta
def dfFormatoPandasTa(df):
    df.index.rename('date', inplace=True)
    df.rename(
    columns={'Abertura': 'open',
             'Máxima': 'high',
             'Mínima' : 'low',
             'Fechamento': 'close',
             'Volume financeiro': 'volume'},
    inplace=True)
    return df

# Computa a estratégia "diamante"
# Entradas:
#   c: preço de fechamento
#   smas: média móvel aritmética de curta duração
#   smal: média móvel artimética de longa duração
def estrategiaDiamante(c, smas, smal):   
    if smas<smal:
        if c < smas:
            reg = 5 # baixa
        elif (c>=smas) and (c<smal):
            reg = 0 # recuperação
        elif c >=smal:
            reg = 1 # acumulação
        else:
            reg = np.NaN # erro
    else:
        if c >= smas:
            reg = 2 # alta
        elif (c>=smal) and (c<smas):
            reg = 3 # aviso
        elif c < smal:
            reg = 4 # distribuição
        else:
            reg = np.NaN # erro
    return reg

def plotaHistograma(D_tr, D_te, n_classes):
    y_tr = D_tr.y.to_numpy()
    y_te = D_te.y.to_numpy()
    hist_tr,_ = np.histogram(y_tr, bins=n_classes)
    hist_te,_ = np.histogram(y_te, bins=n_classes)
    plt.figure()
    plt.title("Treinamento")
    plt.bar(range(n_classes),hist_tr)
    plt.figure()
    plt.title("Teste")
    plt.bar(range(n_classes),hist_te)


# *****************************  PARÂMETROS  ********************************
# Número de experimentos
N = 10

# Estratégia de treinamento
#   False: treinamento em batelada
#   True: Treinamento "On-Line"
onlineTraining = True   

# Base de dados
test_size = 0.3             # tamanho do conjunto de testes (split)
dataInicial = '2009-02-02'  # primeira data desejada com dados completos
dataFinal = '2015-06-30'

# Estratégia "diamante"
p_sm = 50            # número de períodos da média curta
p_lm = 200           # número de períodos da média longa
n_classes = 6        # número de classes da esratégia diamante

# Retina
grau_serie = 5       # Número de periodos passados
n_per_ad = p_lm-1+grau_serie

# Parâmetros da WiSARD
addressSize = 5                 # número de bits de enderaçamento das RAMs
bleachingActivated= True        # desempate
ignoreZero  = False             # RAMs ignoram o endereço 0
completeAddressing = True       # quando M (núm. bits) não é divisível por n_i
verbose = False                 # mensanges durante a execução
returnActivationDegree = False  # retornq o grau de similariedade de cada y
returnConfidence = False        # retorna o grau de confiança de cada y
returnClassesDegrees = False    # confiança de cada y em relação a cada classe


# *************************** 1. PRÉ-PROCESSAMENTO ***************************

#
# 1.1 Importação e tratamento da base de dados
#
df0 = pd.read_csv('datasets/BBDC4_1994-07-04_2020-06-26_Profit.csv',
                  index_col=0, dayfirst=True, parse_dates=True)
# Ordena por datas
df0.sort_index(inplace=True)

# Filtra o período desejado de simulação, levando-se em consideração o 
# período histórico necessário para o cálculo dos dados
delta = datetime.timedelta(days=+10)
data_i = datetime.date.fromisoformat(dataInicial)+delta
df_temp_i = df0[dataInicial:data_i.isoformat()]
idx_i = df_temp_i.iloc[0].name
pos_i = df0.index.get_loc(idx_i)-n_per_ad
data_i2 = df0.iloc[pos_i].name
df0 = df0[data_i2:dataFinal].copy()

#
# 1.2 Aplicação dos indicadores técnicos
#
# Formata o dataframe no formato da biblioteca pandas_ta
dfFormatoPandasTa(df0)

# Calcula as médias móveis e adiciona ao dataframe
smas = df0.ta(kind='sma', length=p_sm, append=False)
df0['smas']=smas
smal = df0.ta(kind='sma', length=p_lm, append=False)
df0['smal']=smal

# Geração dos rótulos conforme a estratégia "diamante"
y = []
for row in df0.itertuples():
    y.append(estrategiaDiamante(row.close, row.smas, row.smal))
    
# Adiciona a nova coluna ao dataframe
df0['y'] = y
y = y[n_per_ad-grau_serie:]

#
# 1.3 Construção da retina e do novo data-set com entradas binarizadas
#
# Retina formada pelos estados dos 5 últimos dias,
# sendo cada dia represetando por uma lista de 6 bits
# região 1 = [1, 0, 0, 0, 0, 0]
# região 2 = [0, 1, 0, 0, 0, 0]
# Xn = [região n-1 | região n-2 | ... | região n-5]


Reg = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
#Reg =  np.identity(n_classes, dtype=int).tolist()

# Gera o conjunto de dados de entrada em formato de listas
X = []
'''
for idx, val in enumerate(y[grau_serie:]):
    x =[]
    for k in range(grau_serie):
        x.extend(Reg[y[idx-k-1]])
    X.append(x)
'''
for idx, val in enumerate(y):
    if idx >= grau_serie:
        x =[]
        for k in range(grau_serie):
            x.extend(Reg[y[idx-k-1]])
        X.append(x)
    
    
# Cria um novo dataset iniciando a partir da primeira linha com dado
# disponíel para a média longa
df = df0.iloc[n_per_ad:].copy()
df['X']=X

X = df.X.tolist()
Y = [round(y) for y in df.y.tolist()]

#
# 1.4 Separação dos conjuntos de treinamento e teste
#
X_tr, X_te, Y_tr1, Y_te1 = train_test_split(X, Y,test_size = test_size, 
                                         shuffle = False)
idx_tr = len(X_tr)
D_tr = df.iloc[:idx_tr].copy()
D_te = df.iloc[idx_tr:].copy()

# transoforma a saída em lista de strings (formato wisardpkg)
Y_tr = list(map(str,Y_tr1))
Y_te = list(map(str,Y_te1))


# ************************** 2. APLICAÇÃO DA WISARD **************************

Acc = np.array([])  # matriz de acurácia
T_tr = np.array([]) # matriz de tempos de treinamento
T_te = np.array([]) # matriz de tempos de classificação

# Número de experimentos
for n in range(N): 
    print("Experimento {0:1d}:".format(n))
    
    # 2.1 Criação do modelo
    wsd = wp.Wisard(addressSize,
                    bleachingActivated = bleachingActivated,
                    ignoreZero = ignoreZero,
                    completeAddressing = completeAddressing,
                    verbose = verbose,
                    returnActivationDegree = returnActivationDegree,
                    returnConfidence = returnConfidence,
                    returnClassesDegrees = returnClassesDegrees)
    
    # 2.2 Treinamento em batelada da base de treinamento
    startTime = time.time()
    wsd.train(X_tr,Y_tr)
    endTime = time.time()
    T_tr_n = endTime-startTime
    print("\tTreinamento concluído em {0:4.2f}ms".format(T_tr_n*1000))
    T_tr = np.append(T_tr,T_tr_n)

    G = [] # lista de saídas preditas
    startTime = time.time()
    
    # 2.3 Classificação com ou sem terinamento on-line
    if onlineTraining:
        # Classificação e treinamento on-line dos exemplares de teste
        for k in range(len(X_te)):
            x = X_te[k]
            g = wsd.classify([x])
            G.append(g[0])
            wsd.train([x],[Y_te[k]])
    else:
        # Classificação em batelada
        G = wsd.classify(X_te)
    
    endTime = time.time()
    T_te_n = endTime-startTime
    print("\tClassificação on-line com treinamento concluídos em {0:4.2f}ms".format(T_te_n*1000))
    T_te = np.append(T_te,T_te_n)

    # ------------------------------ Avaliações  -----------------------------

    # Transformação das listas de string em listas de inteiros 
    G = list(map(int, G))
    Y = list(map(int, Y_te))
    
    # Avalição da acurácia no experimento n
    Acc_n = accuracy_score(Y,G)
    Acc = np.append(Acc, Acc_n)
    print("\tAcurácia {0:1.3f}".format(Acc_n))


# ************************** 3. RESULTADOS **************************

# 3.1 Impressões
# Tempos médios de treinamento e classificação
print("\nTempo médio de treinamento: {0:4.2f}ms".format(T_tr.mean()*1000))
print("Tempo médio de classificação: {0:4.2f}ms".format(T_te.mean()*1000))

# Acurácia média
Acc_mean = Acc.mean()
Acc_std = Acc.std()
print("\nAcurácia média: {0:1.3f} \u00B1 {1:1.3f}".format(Acc.mean(), Acc.std()))

# Matriz de confusão do último experimento
labels = [0, 1, 2, 3, 4, 5]
C = confusion_matrix(Y,G)
print("\nMatriz de confusão do último experimento:")
print("\t Linhas: y")
print("\tColunas: g")
with np.printoptions(precision=2):
    C1 = C.astype(np.float).sum(axis=1)
    print((C.T/C1).T)


# 3.2 Gráfico
plt.figure()

# a) Gráfico de Y
# Separa as regiões em Y
s1 = pd.Series(Y_te1,name='regiao',index=D_te.index)
s = s1.loc[s1.shift() != s1]
s2= s1.loc[s1.last_valid_index()]
s3=pd.Series([s2],index=[s1.last_valid_index()])
s = s.append(s3)

plt.subplot(2,1,1)
D_te.close.plot(linewidth=1, color='black', legend='Fech.')
D_te.smal.plot(linewidth=2, color='blue', legend='SMA 50')
D_te.smas.plot(linewidth=1, color='blue', legend='SMA 100')
plt.title('Real (Y)')
plt.xticks([])
plt.xlabel('')
plt.ylabel('R$')

colors = ['springgreen','mediumseagreen','seagreen',
          'lightcoral','indianred','brown']
pos = 0
for index, value in s.items():
    if index != s.last_valid_index():
        plt.axvspan(index, s.index[pos+1], alpha=0.5, color=colors[value], lw=0)
    pos = pos+1


# b) Gráfico das regiões - G predito
# separa as regiões em G
s1p = pd.Series(G, name='regiao',index=D_te.index)
sp = s1p.loc[s1p.shift() != s1p]
s2p= s1p.loc[s1p.last_valid_index()]
s3p=pd.Series([s2p],index=[s1p.last_valid_index()])
sp = sp.append(s3p)

plt.subplot(2,1,2)
D_te.close.plot(linewidth=1, color='black')
D_te.smal.plot(linewidth=2, color='blue')
D_te.smas.plot(linewidth=1, color='blue')
plt.title('Predição (G)')
plt.xlabel('')
plt.ylabel('R$')

pos = 0
for index, value in sp.items():
    if index != sp.last_valid_index():
        plt.axvspan(index, sp.index[pos+1], 
                    alpha=0.5, color=colors[value], lw=0)
    pos = pos+1