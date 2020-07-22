# -*- coding: utf-8 -*-
"""
Federal University of Rio de Janeiro
Computer Science and Systems Engineering Program (PESC/COPPE)
CPS841 - Weigthless Neural Networks
Prof. Priscila Machado Vieira Lima

Forecasting stock trends with WiSARD
@ide: Spyder
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import pandas_ta as ta # Techical Analysis with Pandas
import time
import wisardpkg as wp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ***************************** 0. PARAMETERS ********************************

# General parameters
N = 1000                          # number of experiments
onlineTraining = True           # online training
validate = True                 # perform validation
plots = False                   # plot the graphics

# Data set loading
stock_cod = 'BBDC4'
filename = 'datasets/' +stock_cod + '.csv'
test_size = 0.2                 # lenght of test dataset
val_size = 0.2                  # length of validation dataset
date_initial = '2017-01-01'      
date_last = '2020-01-31'

# Time series model
T = 5                           # number of periods of time series

# Dataset labeling
h = 3                           # forecast horizon

# Technical indicators used (features) in by WiSARD model
tI = {'vol':    False,
      'trend':  False,
      'ma':     True,
      'bb':     True,
      'macd':   True,
      'rsi':    True,
      'dc':     False,
      'kc':     False,
      'adx':    True,
      'aroon':  False, 
      'mfi':    True,
      }

# Technical indicators parameters
fast_p = 10                      # Cross mean fast period
slow_p = 50                      # Cross mean slow period
bb_per = 20                     # Bollinger bands period
macd_fast_p = 12                # MACD fast period
macd_slow_p = 26               # MACD low period
macd_sign_p = 9                 # MACD signal period
rsi_per = 14                     # RSI period
rsi_ulv = 70                    # RSI upper limit value
rsi_llv = 30                    # RSI lower limit value
dc_llen = 5                    # Donchian Channel lower length
dc_ulen = 5                    # Donchian Channel upper length
kc_len = 20                     # Kelter Channel lenght
adx_len = 10                    # ADX length
adx_ul = 25                     # ADX upper limit
aroon_per = 14                  # Aroon period
aroon_sl = 80                   # Aroon strenght limit
aroon_wl = 40                   # Aroon weakness limit
mfi_per = 5                    # Money Flow Index period
mfi_ul = 80                     # Money Flow Index upper limit
mfi_ll = 20                     # Money Flow Index lower limit

# clusWiSARD
addressSize = 8                 # number of bits for RAMs adressing
minScore = 0.6
threshold = 10
discriminatorLimit = 2

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
    if tI['vol']:
        s_vol = df['volume']
        dfI['vol_cres'] = increasingVol(s_vol)
        dfI['vol_decres'] = decreasingVol(s_vol)
    
    # Upward / downward trend
    if tI['trend']:
        dfI['trend_upw']  = trendUpw(df['close'])
        dfI['trend_dow'] = trendDow(df['close'])
    
    #  Moving average indicators
    if tI['ma']:   
        dfI['ma_buy'], dfI['ma_sell'], dfI['ma_upw'] = movingAverage(df, 
                                                                     fast_p, 
                                                                     slow_p)
    # Bollinger bands
    if tI['bb']:  
        dfI['bb_ll'], dfI['bb_ul'] = bollingerBands(df, bb_per)
    
    # MACD 
    if tI['macd']: 
        dfI['macd_buy'], dfI['macd_sell'], dfI['macd_upw'] = macd(df, 
                                                                  macd_fast_p, 
                                                                  macd_slow_p, 
                                                                  macd_sign_p)
    # RSI
    if tI['rsi']: 
        dfI['rsi_ll'], dfI['rsi_ul'] = rsi(df, rsi_per, rsi_llv, rsi_ulv)
    
    # Donchian Channel
    if tI['dc']:
        dfI['dc_ll'] , dfI['dc_ul'] = donchianChannel(df, dc_llen, dc_ulen)
    
    # Keltner Channel
    if tI['kc']:
        dfI['kc_ll'] , dfI['kc_ul'] = keltnerChannel(df, kc_len)
    
    # ADX (Average Directional Movement Index)
    if tI['adx']:
        adx_force, adx_buy, adx_sell, adx_upw = adx(df, adx_len, adx_ul)
        dfI['adx_force'] = adx_force
        dfI['adx_buy'] =  adx_buy 
        dfI['adx_sell'] = adx_sell
        dfI['adx_upw'] = adx_upw

    # Aroon
    if tI['aroon']:
        aroon_usl, aroon_dsl, aroon_uwl, aroon_dsl = aroon(df, aroon_per, 
                                                           aroon_sl, aroon_wl)
        dfI['aroon_usl'] = aroon_usl
        dfI['aroon_dsl'] = aroon_dsl
        dfI['aroon_uwl'] = aroon_uwl
        dfI['aroon_dwl'] = aroon_dsl
    
    # Money flow Index
    if tI['rsi']:
        dfI['mfi_buy'], dfI['mfi_sell'] = mfi(df, mfi_per, mfi_ll, mfi_ul)
    
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
    adx_buy = ta.cross(dmp, dmm, above=True)
    adx_sell = ta.cross(dmp, dmm, above=False)
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


# ---------------------- Return Calculation  ----------------------

# Buy & Hold return
def bhReturn(df0, D):
    date_i = D.index[0]
    date_f = D.last_valid_index()
    price_i = df0.close[date_i]
    price_f = df0.close[date_f]
    R = price_f/price_i-1
    idx = [date_i, date_f]
    d = [0, 100*R]
    s_r = pd.Series(d,idx)
    return R, s_r

# Simple strategy based on WiSARD forecasts
# Politic: Buy if g=1, hold until g=0
def wsdReturn(df, D, G):
    s1p = pd.Series(G, name='op',index=D.index)
    sp = s1p.loc[s1p.shift() != s1p]
    # s2p= s1p.loc[s1p.last_valid_index()]
    #s3p=pd.Series([s2p],index=[s1p.last_valid_index()])
    #sp = sp.append(s3p)
    R = np.array([])
    index = sp.index.tolist()
    close = sp.values.tolist()
    r_idx=[]   
    for idx, C in enumerate(close):
        if C==0 and idx!=0:
            price = df.close.loc[index[idx]]
            p_price = df.close.loc[index[idx-1]]
            r = price/p_price
            R = np.insert(R,0,r)
            r_idx.append(index[idx])
    R_wsd=R.prod()-1
    s_r = pd.Series(R,r_idx,name='r')
    R_acc = []
    for i,r in enumerate(R):
        if i==0: 
            r_acc = 1
        elif i==1:
            r_acc = r*R[i-1]
        else:
            r_acc = r*R_acc[i-1]
        R_acc.append(r_acc)
    df_R = pd.DataFrame(s_r)
    df_R['r_acc']=R_acc
    return R_wsd, df_R 

# ************************ 2. DATASET PRE-PROCESSING ************************

# DATAFRAMES:
# df0: original dataset with ohlcv (open, high, low, close, volume) data 
# dfI: dataframe with the techincal indicators signals

# Dataset load
df0 = pd.read_csv(filename, index_col=0, dayfirst=True, parse_dates=True)

# Order by date
df0.sort_index(inplace=True)

# Desired period
df0 = df0.loc[date_initial:date_last]

# Format the dataset according to pandas_ta library
pandasTaFormat(df0)

# Construct the indicator dataset formed by tehcnical indicators singals
dfI = signalsConstructor(df0)

# ------------------------------- Labeling ---------------------------------
# Compute the binary trend based on horizon 'h' (days after)
# 0: downward (sell)
# 1: upward (buy)
Y = getBinaryTrend(df0.close, h)

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

tv_size = test_size + val_size
X = D.X.values.tolist()
Y = D.Y.values.tolist()
X_tr, X_tv, Y_tr, Y_tv = train_test_split(X, Y, test_size = tv_size,                            
                                            shuffle = False)

X_te, X_vl, Y_te, Y_vl = train_test_split(X_tv, Y_tv, 
                                          test_size = test_size/tv_size,                            
                                          shuffle = False)

len_tr = len(X_tr)
len_te = len(X_te)

D_tr = D.iloc[:len_tr].copy()
D_te = D.iloc[len_tr:(len_tr+len_te)].copy()
D_vl = D.iloc[len_tr+len_te:].copy()

# maps 'Y to string (wisardpkg format)
Y_tr = list(map(str,Y_tr))
Y_te = list(map(str,Y_te))
Y_vl = list(map(str,Y_vl))


# ************************** 3. WISARD MODEL **************************

Acc_te = np.array([])       # accuracy matrix (test dataset)
Acc_vl = np.array([])       # accuracy matrix (test dataset)
T_tr = np.array([])         # train time matrix
T_te = np.array([])         # classification time matrix (test dataset)
R_wsd_te = np.array([])     # wisard return matrix (test dataset)
R_wsd_vl = np.array([])     # wisard return matrix (test dataset)

for n in range(N): 
    
    # Model initialiation
    wsd = wp.ClusWisard(addressSize,
                    minScore,
                    threshold,
                    discriminatorLimit)
    
    # Batch training
    startTime = time.time()
    wsd.train(X_tr,Y_tr)
    endTime = time.time()
    T_tr_n = endTime-startTime
    T_tr = np.append(T_tr,T_tr_n)

    G_te = [] # predicted output (test dataset)
    startTime = time.time()
    
    # On-line traning and classification
    if onlineTraining:
        for k in range(len(X_te)):
            x = X_te[k]
            g = wsd.classify([x])
            G_te.append(g[0])
            wsd.train([x],[Y_te[k]])
    else:
        # Batch classificaiton
        G_te = wsd.classify(X_te)
    
    endTime = time.time()
    T_te_n = endTime-startTime
    T_te = np.append(T_te,T_te_n)

    # ------------------------------ Evaluation  -----------------------------

    # Map string list to int
    G_te_int = list(map(int, G_te))
    Y_te_int = list(map(int, Y_te))
    
    # Accuracy of experiment n
    Acc_te_n = accuracy_score(Y_te_int,G_te_int)
    Acc_te = np.append(Acc_te, Acc_te_n)
    
    # Return of experiment n
    R_bh_te, s_bh_te = bhReturn(df0,D_te) # Buy and hold return
    R_wsd_te_n, df_R_te = wsdReturn(df0,D_te,G_te_int) # WiSARD returm
    R_wsd_te = np.append(R_wsd_te, R_wsd_te_n)
    

    # ************************** 4. VALIDATION **************************
    if validate:
        G_vl = wsd.classify(X_vl)
        G_vl_int = list(map(int, G_vl))
        Y_vl_int = list(map(int, Y_vl))
        Acc_vl_n = accuracy_score(Y_vl_int,G_vl_int)
        Acc_vl = np.append(Acc_vl, Acc_vl_n)
        
        R_bh_vl, s_bh_vl = bhReturn(df0,D_vl)
        R_wsd_vl_n, df_R_vl = wsdReturn(df0,D_vl,G_vl_int)
        R_wsd_vl = np.append(R_wsd_vl, R_wsd_vl_n)


# ************************** 5. PRINT RESULTS **************************

# Dates split
date_i_tr = D_tr.index[0] # initial trianing date
date_f_tr = D_tr.last_valid_index()# final trianing date
date_i_te = D_te.index[0] # initial testing date
date_f_te = D_te.last_valid_index() # final testing date
date_i_vl = D_vl.index[0] # initial validating date
date_f_vl = D_vl.last_valid_index() # final validating date

print("\n------- TESTING RESULTS --------- ")

# Time performance
print("\nMean time to train: {0:4.2f}ms".format(T_tr.mean()*1000))
print("Mean time to classify: {0:4.2f}ms".format(T_te.mean()*1000))

# Accuracy
print("\nMean accuracy: {0:1.3f} \u00B1 {1:1.3f}".format(Acc_te.mean(), 
                                                         Acc_te.std()))

# Confusion Matrix
C_te = confusion_matrix(Y_te_int,G_te_int)
print("\nConfusion matrix (last experiment):")
print("\tLines: y")
print("\tColumns: g")
with np.printoptions(precision=2):
    C1_te = C_te.astype(np.float).sum(axis=1)
    print((C_te.T/C1_te).T)

# Percentage returns
print("\nPercentage returns (mean):")

# Buy & Hold return
print("\tBuy & Hold: {0:2.2f}%".format(R_bh_te*100))

# Wisard return
#   buy if g=1, hold until g=0
#   op_sig: series with operation signals
n_te = df_R_te.size

print("\tWiSARD: {0:2.2f} \u00B1 {1:1.2f}%".format(R_wsd_te.mean()*100,
                                                   R_wsd_te.std()*100))
print("\t\tPeriod:",date_i_te, "to", date_f_te)
print("\t\tNumber of trades:",n_te)


if validate:
    print("\n------- VALIDATION RESULTS -------- ")
    
    # Accuracy
    print("\nMean accuracy: {0:1.3f} \u00B1 {1:1.3f}".format(Acc_vl.mean(), 
                                                             Acc_vl.std()))
    
    # Confusion Matrix 
    C_vl = confusion_matrix(Y_vl_int,G_vl_int)
    print("\nConfusion matrix (last experiment):")
    print("\tLines: y")
    print("\tColumns: g")
    with np.printoptions(precision=2):
        C1_vl = C_vl.astype(np.float).sum(axis=1)
        print((C_vl.T/C1_vl).T)

    # Percentage returns
    print("\nPercentage returns (mean):")
    
    # Buy & Hold return
    print("\tBuy & Hold: {0:2.2f}%".format(R_bh_vl*100))
    
    # Wisard return
    #   buy if g=1, hold until g=0
    #   op_sig: series with operation signals
    n_vl = df_R_vl.size
    print("\tWiSARD: {0:2.2f} \u00B1 {1:1.2f}%".format(R_wsd_vl.mean()*100,
                                                       R_wsd_vl.std()*100))
    print("\t\tPeriod:",date_i_vl, "to", date_f_vl)
    print("\t\tNumber of operations:",n_vl)

# ------------------------------ Plots  -----------------------------

if plots:
    # plot close price of training+testing+validating period
    fig,ax=plt.subplots()
    date_f = D_vl.last_valid_index()
    
    df0.loc[date_i_tr:date_f].close.plot(ax=ax)
    ax.set_title(stock_cod)
    ax.set_xlabel("")
    ax.set_ylabel("Price (R$)")
    
    # plot training, testing and validate regions
    ax.axvspan(date_i_tr, date_f_tr, alpha=0.5, color='gray', lw=0)
    ax.axvspan(date_i_te, date_f_te, alpha=0.3, color='gray', lw=0)
    ax.axvspan(date_i_vl, date_f_vl, alpha=0.1, color='gray', lw=0)
    tr_pt = mpatches.Patch(color='gray', alpha=0.5, label="Train")
    te_pt = mpatches.Patch(color='gray', alpha=0.3, label="Test")
    vl_pt = mpatches.Patch(color='gray', alpha=0.1, label="Validation")
    
    # plot test and validating returns
    l2 = r_wsd_te_acc_s = (df_R_te['r_acc']-1)*100
    r_wsd_te_acc_s.plot(ax=ax, secondary_y=True, linewidth=1, color='C1', label="WiSARD")
    s_bh_te.plot(ax=ax, secondary_y=True, linewidth=1, color='C2', style='-', label="B&H")
    
    if validate:
        r_wsd_vl_acc_s = (df_R_vl['r_acc']-1)*100
        r_wsd_vl_acc_s.plot(ax=ax, secondary_y=True, linewidth=1, color='C1')
        s_bh_vl.plot(ax=ax, secondary_y=True, linewidth=1, color='C2', style='-')
    
    ax.right_ax.set_ylabel("Acc. Return (%)")
    ax.right_ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f'))
    ax.legend(handles=[tr_pt, te_pt, vl_pt],loc='upper left')
    
    colors = ['C0', 'C1', 'C2']
    lines = [Line2D([0], [0], color=c) for c in colors]
    labels=['Close','WSD return', 'B&H return']
    plt.legend(lines,labels,loc='upper left', bbox_to_anchor=(0,0.75))
