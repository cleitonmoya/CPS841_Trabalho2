# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
Porgrama de Pos-Graduação em Engenharia de Sistemas e Computação
CPS841 - Redes Neurais Sem Peso

Teste da WiSARD para predição de tendências de ações com a estratégia 
"diamante, conforme o artigo "Análise de Séries Temporais Financeiras 
Utilizando WiSARD" de Samara Alves

@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

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

# Estratégia "diamante"
def determinaRegiao(c, smas, smal):   
    if smas<smal:
        if c < smas:
            reg = 0 # baixa
        elif (c>=smas) and (c<smal):
            reg = 1 # recuperação
        elif c >=smal:
            reg = 2 # acumulação
        else:
            reg = -1
    else:
        if c >= smas:
            reg = 3 # alta
        elif (c>=smal) and (c<smas):
            reg = 4 # aviso
        elif c < smal:
            reg = 5 #distribuição
        else:
            reg = -2 # erro
    return reg


# **************************** PRÉ-PROCESSAMENTO ****************************

# Importa a base de dados
df0 = pd.read_csv('datasets/BBDC4_2008-09-08_2015-06-30.csv',index_col=0)

# Formata o dataframe no formato da biblioteca pandas_ta
dfFormatoPandasTa(df0)

# Calcula as médias móveis
sma50 = df0.ta(kind='sma', length=50, append=True)
sma100 = df0.ta(kind='sma', length=100, append=True)

# Mapeia as regiões conforme a estratégia "diamante"
est_dia = []
for row in df0.itertuples():
    est_dia.append(determinaRegiao(row.close, row.SMA_50, row.SMA_100))
    
# Adiciona a coluna com a estratégia
df0['est_dia'] = est_dia

# Cria um novo data-set iniciando em 2009-02-02
df = df0.iloc[100:].copy()

# Gera o dataset binarizado com a retina formada pelos 5 últimos dias,
# sendo cada dia represetando por uma lista de 6 bits
# região 1 = [1, 0, 0, 0, 0, 0]
# região 2 = [0, 1, 0, 0, 0, 0]
# Xn = [região n-1 | região n-2 | ... | região n-5]
est_dia = est_dia[100:]
Reg =  [[1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]]

# Gera o conjunto de dados de entrada em formato de listas
X = []
for idx, val in enumerate(est_dia[5:]):
    x =[]
    for k in range(5):
        x.extend(Reg[est_dia[idx-k-1]])
    X.append(x)
y = est_dia[5:]

# Transforma y em uma lista de strings (formato wisardpkg)
y = list(map(str, y))


# *************************** APLICAÇÃO DA WISARD ***************************



'''
# Filtra as regiões por datas
s = pd.Series(reg[100:],name='regiao',index=df2.index)
s = s.loc[s.shift() != s]

# Plota os gráficos
df2.close.plot(linewidth=0.5)
df2.SMA_50.plot(linewidth=0.75)
df2.SMA_100.plot(linewidth=0.75)

colors = ['red','orange','yellow','green','lime','yellowgreen']
pos = 0
for index, value in s.items():
    if index != s.last_valid_index():
        plt.axvspan(index, s.index[pos+1], alpha=0.1, color=colors[value-1], lw=0)
    pos = pos+1
'''