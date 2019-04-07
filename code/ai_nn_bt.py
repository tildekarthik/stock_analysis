#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:07:04 2018

@author: karthik
"""
import pandas as pd
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import RandomForestRegressor
from stock_lib import *




def main():
    F_DAYS=1
    bt_days=1
    col_predict='Close'
    out_l = []
    #For AI
    BATCH_SIZE=32
    EPOCHS=32

    stk_list = ['HINDUNILVR']
#    stk_list=['NIFTY_50','BANKNIFTY','HINDUNILVR','TITAN']
#    stk_list=['NIFTY_50','BANKNIFTY','HINDUNILVR','TITAN','ASIANPAINT','HDFCBANK','GRASIM','GLAXO','AUROPHARMA','TITAN','ICICIBANK']
#    stk_list=['ASIANPAINT','AUROPHARMA','COLPAL','CADILAHC','FINCABLES','FEDERALBNK','GLAXO',
#                'GRASIM','HDFCBANK','ICICIBANK','INFY','ITC','KOTAKBANK','TATAMOTORS','RELIANCE',
#                'HINDUNILVR','HINDALCO','LT','MARICO','SBIN','SUNPHARMA','TATACHEM',
#                'TATASTEEL','TCS','TITAN']    

    for stk in stk_list:
        print("Processing:"+stk)
        df_main = read_stk_data(stk,2003,2019)
        df_main = cleanse_data(df_main,stk)
        df_main = prepare_data(df_main)
        
        
        out=df_main[['Symbol',col_predict]].copy()
        out['y']=out[col_predict].shift(-F_DAYS)
        
        
        #mod = GradientBoostingRegressor(n_estimators=96)
        #mod = RandomForestRegressor(n_estimators=200)
        
        out=back_test_AI_NSE(bt_days,df_main,F_DAYS,col_predict,out,BATCH_SIZE,EPOCHS)
        out_l.append(out.dropna(axis=0))
        out.dropna(axis=0).to_csv(stk+'_NSEAnalysisv3.csv')
    out = pd.concat(out_l,axis=0)
    out.to_csv('stocks.csv')
    return out


if __name__=='__main__':
    out = main()
    #out.dropna(axis=0)[['Close','y','pred']].plot()

