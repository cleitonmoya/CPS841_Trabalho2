# -*- coding: utf-8 -*-
"""
Federal Uniersity of Rio de Janeiro
Computer Science and Systems Engineering Program (PESC/COPPE)
CPS841 - Weigthless Neural Networks
Prof. Priscila Machado Vieira Lima

Forecasting stock price trends with WiSARD

@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import time
import wisardpkg as wp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ***************************** 0. PARAMETERS ********************************

# General parameters
N = 5                           # number of experiments
onlineTraining = True           # online traiing

# Data set loading
test_size = 0.1                 # lenght of est datasize
date_initial = '2015-01-01'      
date_last = '2020-01-01'

# Technical indicators
rsi_ulv = 70                    # RSI upper limit value
rsi_llv = 30                    # RSI lower limit value
dc_llen = 20                    # Donchian Channel lower length
dc_ulen = 20                    # Donchian Channel upper length
kc_len = 20                     # Kelter Channel lenght
adx_len = 14                    # ADX length
adx_ul = 25                     # ADX upper limit
aroon_per = 14                  # Aroon period
aroon_sl = 70                   # Aroon strenght limit
aroon_wl = 50                   # Aroon weakness limit
mfi_per = 14                    # Money Flow Index period
mfi_ul = 80                     # Money Flow Index upper limit
mfi_ll = 20                     # Money Flow Index lower limit

# WiSARD
bleachingActivated= True        # bleaching mechanism
ignoreZero  = False             # RAMs ignore adress 0
completeAddressing = True       # when M is not divided by n_i
verbose = False                 # print status messages
returnActivationDegree = False  # print similarity of each y
returnConfidence = False        # print confidence of each y
returnClassesDegrees = False    # print confidente of each y per classes


# ************************** 1. AUXILIARY FUNCTIONS **************************

# ---------------------- Dataset Construction ----------------------

# Format the dataframe according to pandas_ta library
def pandasTaFormat(df):
    df.index.rename('date', inplace=True)
    df.rename(
        columns={'Abertura': 'open',
                 'Máxima': 'high',
                 'Mínima' : 'low',
                 'Fechamento': 'close',
                 'Volume Financeiro': 'volume'},
        inplace=True)

# Construct a datframe with the indicators signals
def signalsConstructor(df):
    
    dfI = pd.DataFrame(index=df.index)
    
    # Increasing / decreasing volume
    s_vol = df['volume']
    dfI['vol_cres'] = increasingVol(s_vol)
    dfI['vol_decres'] = decreasingVol(s_vol)
    
    # Upward / downward trend
    dfI['trend_upw']  = trendUpw(df['close'])
    dfI['trend_dow'] = trendDow(df['close'])
    
    # Moving average indicators
    dfI['ma_buy'], dfI['ma_sell'], dfI['ma_upw'] = movingAverage(df, fast_p, 
                                                                 slow_p)
    
    # Bollinger bands
    dfI['bb_ll'], dfI['bb_ul'] = bollingerBands(df, bb_per)
    
    # MACD 
    dfI['macd_buy'], dfI['macd_sell'], dfI['macd_upw'] = macd(df, 
                                                              macd_fast_p, 
                                                              macd_slow_p, 
                                                              macd_sign_p)
    # RSI
    dfI['rsi_ll'], dfI['rsi_ul'] = rsi(df, rsi_per, rsi_llv, rsi_ulv)
    
    '''
    # Donchian Channel
    dfI['dc_ll'] , dfI['dc_ul'] = donchianChannel(df, dc_llen, dc_ulen)
    
    # Keltner Channel
    dfI['kc_ll'] , dfI['kc_ul'] = keltnerChannel(df, kc_len)
    
    # ADX (Average Directional Movement Index)
    adx_force, adx_buy, adx_sell, adx_upw = adx(df, adx_len, adx_ul)
    dfI['adx_force'] = adx_force
    dfI['adx_buy'] =  adx_buy 
    dfI['adx_sell'] = adx_sell
    dfI['adx_upw'] = adx_upw

    # Aroon
    aroon_usl, aroon_dsl, aroon_uwl, aroon_dsl = aroon(df, aroon_per, 
                                                       aroon_sl, aroon_wl)
    dfI['aroon_usl'] = aroon_usl
    dfI['aroon_dsl'] = aroon_dsl
    dfI['aroon_uwl'] = aroon_uwl
    dfI['aroon_dwl'] = aroon_dsl
    
    # Money flow Index
    dfI['mfi_buy'], dfI['mfi_sell'] = mfi(df, mfi_per, mfi_ll, mfi_ul)
    '''
    return dfI

# s_close: Pandas Series with close prices 
def getBinaryTrend(s_close, h):
    c = s_close.tolist()
    Y = []
    
    # Binary trend classification
        # 0: downward trend
        # 1: upward trend 
    for k, val in enumerate(c):
        if k<len(c)-h:
            if c[k+h] > val:
                y = 1
            else:
                y = 0
            Y.append(y)
        else:
            Y.append(np.nan)
    return Y

# get the input matrix with lags
# T: number of lag periods
# x(t) = x(t) + x(t-1) + x(t-2) + .. +x(t-T) 
def timeSeriesModel(X1,T):
    X2 = []
    for idx, val in enumerate(X1):
        if idx >= T:
            x = []
            for t in range(T):
                x.extend(X1[idx-t])
            X2.append(x)
        else:
            X2.append(np.nan)
    return X2

# ---------------------- Technical indicators signals ----------------------

# Increasing financial volume
def increasingVol(s_vol):
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

# Decreasing financial volume
def decreasingVol(s_vol):
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

# Upward trending based on close price 
def trendUpw(s_close):
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

# Downward trending based on close price 
def trendDow(s_close):
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

# Moving averages cross signals
def movingAverage(df, fast_p, slow_p):
    smas = df.ta.sma(length=fast_p)
    smal = df.ta.sma(length=slow_p)
    
    ma_buy = ta.cross(smas, smal, above=True) 
    ma_sell = ta.cross(smas, smal, above=False) 
    ma_upw = ta.above(smas, smal)
    
    ma_buy.iloc[:fast_p] = np.nan
    ma_sell.iloc[:fast_p] = np.nan
    ma_upw.iloc[:fast_p] = np.nan
    
    return ma_buy, ma_sell, ma_upw

def bollingerBands(df, bb_per):
    df_bb = df.ta.bbands(length=bb_per) 
    bol_low = df_bb.iloc[:,0]
    bol_upp = df_bb.iloc[:,2]
    
    bb_ll = ta.below(df.close, bol_low)
    bb_ul = ta.above(df.close, bol_upp)
    
    bb_ll.iloc[:bb_per] = np.nan
    bb_ul.iloc[:bb_per] = np.nan
    
    return bb_ll, bb_ul

def macd(df, macd_fast_p, macd_slow_p, macd_sign_p):
    df_macd = df.ta.macd(fast=macd_fast_p, slow=macd_slow_p, 
                      signal=macd_sign_p)
    macd_macd = df_macd.iloc[:,0]
    macd_sign = df_macd.iloc[:,2]
    
    macd_buy = ta.cross(macd_sign, macd_macd, above=True)
    macd_sell = ta.cross(macd_sign, macd_macd, above=False)
    macd_upw = ta.above(macd_sign, macd_macd)
    
    macd_buy.iloc[:macd_slow_p] = np.nan
    macd_sell.iloc[:macd_slow_p] = np.nan
    macd_upw.iloc[:macd_slow_p] = np.nan
    
    return macd_buy, macd_sell, macd_upw

def rsi(df, rsi_per, rsi_llv, rsi_ulv):
    s_rsi = df.ta.rsi(length=rsi_per)
    s_rsi.iloc[:rsi_per] = np.nan # ta includes first values
   
    rsi_ll = ta.below_value(s_rsi, value=rsi_llv)
    rsi_ul = ta.above_value(s_rsi, value=rsi_ulv)
   
    rsi_ll.iloc[:rsi_per] = np.nan
    rsi_ul.iloc[:rsi_per] = np.nan
    
    return rsi_ll, rsi_ul

def donchianChannel(df, dc_llen, dc_ulen):
    df_dc = ta.donchian(df.close, uper_lenght=dc_ulen, lower_length=dc_llen)
    dc_low = df_dc.iloc[:,0]
    dc_upp = df_dc.iloc[:,2]
    
    dc_ul = ta.above(df.close, dc_upp)
    dc_ll = ta.below(df.close, dc_low)
    
    dc_ul.iloc[:max([dc_llen, dc_ulen])] = np.nan
    dc_ll.iloc[:max([dc_llen, dc_ulen])] = np.nan
    
    return dc_ll, dc_ul
        
def keltnerChannel(df, kc_len):
    df_kc = df.ta.kc(length=kc_len)
    kc_low = df_kc.iloc[:,0]
    kc_upp = df_kc.iloc[:,2]
    
    kc_ul = ta.above(df.close, kc_upp)
    kc_ll = ta.below(df.close, kc_low)
    
    kc_ul.iloc[:kc_len] = np.nan
    kc_ll.iloc[:kc_len] = np.nan
    
    return kc_ll, kc_ul 

def adx(df, adx_len, adx_ul):
    df_adx = df.ta.adx(length=adx_len)
    adx, dmp, dmm = df_adx.iloc[:,0], df_adx.iloc[:,1], df_adx.iloc[:,2]
    
    adx_force = ta.above_value(adx,adx_ul)
    adx_buy = ta.cross(dmm, dmp, above=True)
    adx_sell = ta.cross(dmm, dmp, above=False)
    adx_upw = ta.above(dmp, dmm)
    
    adx_force.iloc[:adx_len] = np.nan
    adx_buy.iloc[:adx_len] = np.nan
    adx_sell.iloc[:adx_len] = np.nan
    adx_upw.iloc[:adx_len] = np.nan
    
    return adx_force, adx_buy, adx_sell, adx_upw

def aroon(df, aroon_per, aroon_sl, aroon_wl):
    df_aroon = df.ta.aroon(length=aroon_per)
    aroon_u, aroon_d = df_aroon.iloc[:,0], df_aroon.iloc[:,1]
    
    aroon_usl = ta.above_value(aroon_u, aroon_sl) # strenght in aroon up
    aroon_dsl = ta.above_value(aroon_d, aroon_sl) # strenght in aroon down
    aroon_uwl = ta.below_value(aroon_u, aroon_wl) # weakness in aroon up
    aroon_dwl = ta.below_value(aroon_d, aroon_wl) # weakness in aroon down
    
    aroon_usl.iloc[:aroon_per] = np.nan
    aroon_dsl.iloc[:aroon_per] = np.nan
    aroon_uwl.iloc[:aroon_per] = np.nan
    aroon_dwl.iloc[:aroon_per] = np.nan
    
    return aroon_usl, aroon_dsl, aroon_uwl, aroon_dwl
  
def mfi(df, mfi_per, mfi_ll, mfi_ul):
    mfi = df.ta.mfi(length=mfi_per)
    mfi_buy = ta.above_value(mfi, mfi_ul)
    mfi_sell = ta.below_value(mfi, mfi_ll)    
    mfi_buy.iloc[:mfi_per] = np.nan
    mfi_sell.iloc[:mfi_per] = np.nan
    return mfi_buy, mfi_sell
    
# ************************ 2. DATASET PRE-PROCESSING ************************

# DATAFRAMES:
# df0: original dataset with ohlcv (open, high, low, close, volume) data 
# dfI: dataframe with the techincal indicators signals

# Dataset load
df0 = pd.read_csv('datasets/BBDC4.csv',
                  index_col=0, dayfirst=True, parse_dates=True)

# Order by date
df0.sort_index(inplace=True)

# Desired period
df0 = df0.loc[date_initial:date_last]

# Format the dataset according to pandas_ta library
pandasTaFormat(df0)


# ------------------------------- Grid Search -------------------------------

addressSizeG = [4, 8, 16]    # number of bits for RAMs adressing 4

# Time series model
TG = [1, 3, 5]           # number of periods of time series 4

# Dataset labeling
hG = [1, 3, 5]            # forecast horizon 4

# Technical indicators
fast_pG = [10, 20, 50]        # Cross mean fast period 3
slow_pG = [50, 100, 200]     # Cross mean slow period 3
bb_perG = [20, 50, 100]       # Bollinger bands period 3
macd_fast_pG = [5, 12, 20]       # MACD fast period 3
macd_slow_pG = [12, 26, 50]      # MACD low period 3
macd_sign_pG = [5, 9, 15]        # MACD signal period 3
rsi_perG = [5, 14, 20]           # RSI period 3
df_res=pd.DataFrame(columns=['addressSize', 'T', 'h',
                             'fast_p','slow_p', 'bb_per',
                             'macd_fast_p', 'macd_slow_p', 'macd_sign_p',
                             'rsi_per', 'Acc'
                             ])

print("Number of combinations:", 3**10)
print("Estimated time:", (3**10)/1000/60, "min")

cont = 0
for k1 in range(len(addressSizeG)):
    addressSize=addressSizeG[k1]
    for k2 in range(len(TG)):
        T = TG[k2]
        for k3 in range(len(hG)):
            h = hG[k3]
            for k4 in range(len(fast_pG)):
                fast_p = fast_pG[k4]
                for k5 in range(len(slow_pG)):
                    slow_p=slow_pG[k5]
                    for k6 in range(len(bb_perG)):
                        bb_per=bb_perG[k6]
                        for k7 in range(len(macd_fast_pG)):
                            macd_fast_p=macd_fast_pG[k7]
                            for k8 in range(len(macd_slow_pG)):
                                macd_slow_p=macd_slow_pG[k8]
                                for k9 in range(len(macd_sign_pG)):
                                    macd_sign_p = macd_sign_pG[k9]
                                    for k10 in range(len(rsi_perG)):
                                        rsi_per=rsi_perG [k10] 
                                        # ------------------------------- Labeling ---------------------------------
                                        # Compute the binary trend based on horizon 'h' (days after)
                                        # 0: downward (sell)
                                        # 1: upward (buy)
                                        Y = getBinaryTrend(df0.close, h)
    
                                        # ------------------------------- Indicators ---------------------------------
                                        # Construct the indicator dataset formed by tehcnical indicators singals
                                        dfI = signalsConstructor(df0)
                                        
                                        # Remove 'nan' lines of dfI and Y  and joint them in new dataset (D1)
                                        D1 = dfI.copy()
                                        D1['Y']=Y
                                        D1 = D1.dropna().astype(int)
                                        
                                        # ------------------------- Time series model ------------------------------
                                        
                                        # Create time-series model with T lag periods
                                        X1 = D1.iloc[:,0:len(D1.columns)-1].values.tolist()
                                        if T>1:
                                            X = timeSeriesModel(X1, T)
                                        else:
                                            X = X1
                                        
                                        # Remove 'nan' from data and create a new dataset with X and Y
                                        Y = D1.iloc[:,len(D1.columns)-1].tolist()
                                        d = {'X': X, 'Y': Y}
                                        D = pd.DataFrame(data=d, index=D1.index).dropna()
                                        
                                        
                                        # -------------------- Dataset splitting -------------------------
                                        
                                        # Split dataset for training and test
                                        X = D.X.values.tolist()
                                        Y = D.Y.values.tolist()
                                        X_tr, X_te, Y_tr1, Y_te1 = train_test_split(X, Y, test_size = test_size,                            
                                                                                    shuffle = False)
                                        idx_tr = len(X_tr)
                                        D_tr = D.iloc[:idx_tr].copy()
                                        D_te = D.iloc[idx_tr:].copy()
                                        
                                        # maps 'Y to string (wisardpkg format)
                                        Y_tr = list(map(str,Y_tr1))
                                        Y_te = list(map(str,Y_te1))
                                        
                                        
                                        # ************************** 3. WISARD MODEL **************************
                                        
                                        Acc = np.array([])  # accuracy matrix
                                        T_tr = np.array([]) # train time matrix
                                        T_te = np.array([]) # classification time matrix
                                        
                                        for n in range(N): 
                                            
                                            # 2.1 Model initialiation
                                            wsd = wp.Wisard(addressSize,
                                                            bleachingActivated = bleachingActivated,
                                                            ignoreZero = ignoreZero,
                                                            completeAddressing = completeAddressing,
                                                            verbose = verbose,
                                                            returnActivationDegree = returnActivationDegree,
                                                            returnConfidence = returnConfidence,
                                                            returnClassesDegrees = returnClassesDegrees)
                                            
                                            # 2.2 Batch training
                                            wsd.train(X_tr,Y_tr)
                                            
                                            
                                            G = [] # lista de saídas preditas
                                            
                                            
                                            # 2.3 Classificação com ou sem terinamento on-line
                                            if onlineTraining:
                                                # Online training and classification
                                                for k in range(len(X_te)):
                                                    x = X_te[k]
                                                    g = wsd.classify([x])
                                                    G.append(g[0])
                                                    wsd.train([x],[Y_te[k]])
                                            else:
                                                # Batch classificaiton
                                                G = wsd.classify(X_te)
                                            
                                            # ------------------------------ Evaluation  -----------------------------
                                        
                                            # Map string list to int
                                            G = list(map(int, G))
                                            Y = list(map(int, Y_te))
                                            
                                            # Accuracy of experiment n
                                            Acc_n = accuracy_score(Y,G)
                                            Acc = np.append(Acc, Acc_n)
                                        
                                        
                                        # ************************** 4. RESULTS **************************
                                        
                                        Acc_mean = Acc.mean()
                                        Acc_std = Acc.std()
                                        d = {'addressSize':addressSize, 
                                             'T':T, 
                                             'h':h,
                                             'fast_p':fast_p,
                                             'slow_p':slow_p, 
                                             'bb_per':bb_per,
                                             'macd_fast_p':macd_fast_p, 
                                             'macd_slow_p':macd_slow_p, 
                                             'macd_sign_p':macd_sign_p,
                                             'rsi_per':rsi_per, 
                                             'Acc':Acc_mean}
                                        df_res =df_res.append(d,ignore_index=True)
                                        cont=cont+1
                                        if cont%10000 == 0: print("Iteração:", cont)
                                        
df_res.to_csv('resultadosGrid.csv')