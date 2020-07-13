# -*- coding: utf-8 -*-
"""
Universidade Federal do Rio de Janeiro
Porgrama de Pos-Graduação em Engenharia de Sistemas e Computação
CPS841 - Redes Neurais Sem Peso

Teste da WiSARD para predição de tendências de ações com a estratégia 
"diamante, conforme o artigo "Análise de Séries Temporais Financeiras 
Utilizando WiSARD"

Gera o dataset e rotula os dados com base nos diversos indicadores técnicos

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
             'Volume Financeiro': 'volume'},
    inplace=True)

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

def volumeCrescente(s_vol):
    vol = s_vol.to_numpy()
    vol_cres = []
    for k, val in enumerate(vol):
        if k>=2:
            if val > 1.1*vol[k-1] and vol[k-1] > 1.1*vol[k-2]:
                vol_cres.append(1)
            else: 
                vol_cres.append(0)
        else:
            vol_cres.append(np.nan)
    return pd.Series(vol_cres, s_vol.index)


def volumeDecrescente(s_vol):
    vol = s_vol.to_numpy()
    vol_decres = []
    for k, val in enumerate(vol):
        if k>=2:
            if val < 0.9*vol[k-1] and vol[k-1] < 0.9*vol[k-2]:
                vol_decres.append(1)
            else: 
                vol_decres.append(0)
        else:
            vol_decres.append(np.nan)
    return pd.Series(vol_decres, s_vol.index)

def tendenciaAlta(s_close):
    close = s_close.to_numpy()
    tend = []
    for k, val in enumerate(close):
        if k>=2:
            if val > 1.005*close[k-1] and close[k-1] > 1.005*close[k-2]:
                tend.append(1)
            else: 
                tend.append(0)
        else:
            tend.append(np.nan)
    return pd.Series(tend, s_close.index)

def tendenciaBaixa(s_close):
    close = s_close.to_numpy()
    tend = []
    for k, val in enumerate(close):
        if k>=2:
            if val < 0.995*close[k-1] and close[k-1] < 0.995*close[k-2]:
                tend.append(1)
            else: 
                tend.append(0)
        else:
            tend.append(np.nan)
    return pd.Series(tend, s_close.index)

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

# Paramétros das etrat
p_sm = 10            # número de períodos da média curta
p_lm = 20           # número de períodos da média longa
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

# 1.2.1 Rastreadores de tendência

# Formata o dataframe no formato da biblioteca pandas_ta
dfFormatoPandasTa(df0)
df1 = pd.DataFrame(index=df0.index)

# Médias móveis
smas = df0.ta.sma(length=p_sm)
df0['smas']=smas
smal = df0.ta.sma(length=p_lm)
df0['smal']=smal
mm_buy = ta.cross(smas, smal, above=True) 
mm_sell = ta.cross(smas, smal, above=False) 
mm_alta = ta.above(smas, smal)

mm_buy.iloc[:p_lm] = np.nan
mm_sell.iloc[:p_lm] = np.nan
mm_alta.iloc[:p_lm] = np.nan

df1['mm_buy'] = mm_buy
df1['mm_sell'] = mm_sell
df1['mm_alta'] = mm_alta

# Bandas de Bollinger
bb_per = 20 # period
df_bb = df0.ta.bbands(length=bb_per) 
df0['bol_low'] = df_bb.iloc[:,0]
df0['bol_upp'] = df_bb.iloc[:,2]
bb_ul = ta.above(df0.close, df0.bol_upp)
bb_ll = ta.below(df0.close, df0.bol_low)
bb_ll.iloc[:bb_per] = np.nan
bb_ul.iloc[:bb_per] = np.nan
df1['bb_ul'] = bb_ul
df1['bb_ll'] = bb_ll

# MACD
macd_fast_per = 12
macd_slow_per = 26
macd_sign_per = 9
df_macd = df0.ta.macd(fast=macd_fast_per, slow=macd_slow_per, 
                  signal=macd_sign_per)
df0['macd_macd'] = df_macd.iloc[:,0]
df0['macd_hist'] = df_macd.iloc[:,1]
df0['macd_sign'] = df_macd.iloc[:,2]

macd_buy = ta.cross(df0['macd_sign'], df0['macd_macd'], above=True)
macd_sell = ta.cross(df0['macd_sign'], df0['macd_macd'], above=False)
macd_alta = mm_alta = ta.above(df0.macd_sign, df0.macd_macd)
macd_alta.iloc[:macd_slow_per] = np.nan
 
df1['macd_buy'] = macd_buy
df1['macd_sell'] = macd_sell
df1['macd_alta'] = macd_alta

# RSI
rsi_per = 14 # period
rsi_ulv = 70 # upper limit value
rsi_llv = 30 # lower limit value

s_rsi = df0.ta.rsi(length=rsi_per)
s_rsi.iloc[:rsi_per] = np.nan # ta includes first values
df0['rsi'] = s_rsi

rsi_ul = ta.above_value(s_rsi, value=rsi_ulv)
df1['bb_ul'] = bb_ul
rsi_ll = ta.below_value(s_rsi, value=rsi_llv)
rsi_ll.iloc[:rsi_per] = np.nan
rsi_ul.iloc[:rsi_per] = np.nan
df1['rsi_ul'] = rsi_ul
df1['rsi_ll'] = rsi_ll

# Volume crescente / decrescente
s_vol = df0['volume']
vol_cres = volumeCrescente(s_vol)
vol_decres = volumeDecrescente(s_vol)
df1['vol_cres'] = vol_cres
df1['vol_decres'] = vol_decres

# tendência de alta / baixa
tendAlta = tendenciaAlta(df0['close'])
tendBaixa = tendenciaBaixa(df0['close'])
df1['tend_alta'] = tendAlta
df1['tend_baixa'] = tendBaixa

# Canal Donchian

# Canal Keltner

# Parabólico SAR

# Movimento direcional

# True Range



# 1.2.2 Momentum



# 1.2.3 Bandas


'''
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
for idx, val in enumerate(y[grau_serie:]):
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
'''



