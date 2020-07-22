#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:33:01 2020

@author: cleiton
"""


filename = 'datasets/PETR4.csv'
date_initial = '2019-02-25'      
date_last = '2020-01-28'

import pandas as pd

# Dataset load
df0 = pd.read_csv(filename, index_col=0, dayfirst=True, parse_dates=True)

# Order by date
df0.sort_index(inplace=True)

# Desired period
df0 = df0.loc[date_initial:date_last]