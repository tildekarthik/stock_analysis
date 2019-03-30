# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import matplotlib.pyplot as plt
import pandas as pd
from nsepy import get_history
from datetime import date
from stock_lib import *

co_name_list = read_yaml('../stocks.yaml')

# indexes
index_list = read_yaml('../indexes.yaml')


# repeat the below code by changing the year to download the data

year = 2019
for stk in co_name_list:
    print('Processing:'+stk)
    stk=stk.replace('NSE/','')
    st = date(year,1,1)
    end = date(year,12,31)
    df = get_history(symbol=stk, start=st, end=end)
    df.to_hdf('nsepy_'+str(year)+'.hdf5',key=stk,mode='a',append=False,index=True)

for stk in index_list:
    print('Processing:'+stk)
    st = date(year,1,1)
    end = date(year,12,31)
    df = get_history(symbol=stk, start=st, end=end, index=True)
    df.to_hdf('nsepy_'+str(year)+'.hdf5',key=stk.replace(' ','_'),mode='a',append=False)

# For checking #
stk="NIFTY_50"
year=2017

l = []
for i in [2017,2018,2019]:
    l.append(pd.read_hdf('nsepy_'+str(i)+'.hdf5',key=stk.replace(' ','_')))

#df_read = pd.read_hdf('nsepy_'+str(year)+'.hdf5',key=stk.replace(' ','_'))
df_read=pd.concat(l,axis=0)


read_stk_data('NIFTY_50',2018,2019)




stock_close_NSE(co_name_list+index_list,2003,2019)