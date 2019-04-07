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
    bt_days=30
    col_predict='Close'
    out_l = []
    #For AI
    EPOCHS=8
    BATCH_SIZE=32
    eval_dict = {
                1:['NIFTY_50',8,32],

                }
    #opt_val={'NIFTY_50':['NIFTY_50',16,16]}
#    eval_dict = {
#            1:['NIFTY_50',16,16],
#            2:['BANKNIFTY',32,32],
#            3:['HINDUNILVR',32,32],
#            4:['TCS',8,32],
#            5:['GRASIM',64,32]
#            }
#    stk_list = ['BANKNIFTY']
#    stk_list=['NIFTY_50','BANKNIFTY','HINDUNILVR','TITAN']
#    stk_list=['NIFTY_50','BANKNIFTY','HINDUNILVR','TITAN','ASIANPAINT','HDFCBANK','GRASIM','GLAXO','AUROPHARMA','TITAN','ICICIBANK']
#    stk_list=['ASIANPAINT','AUROPHARMA','COLPAL','CADILAHC','FINCABLES','FEDERALBNK','GLAXO',
#                'GRASIM','HDFCBANK','ICICIBANK','INFY','ITC','KOTAKBANK','TATAMOTORS','RELIANCE',
#                'HINDUNILVR','HINDALCO','LT','MARICO','SBIN','SUNPHARMA','TATACHEM',
#                'TATASTEEL','TCS','TITAN']    

    for iter in eval_dict.keys():
        print("Processing:"+str(iter))
        stk,EPOCHS,BATCH_SIZE=eval_dict[iter]

        df_main = read_stk_data(stk,2003,2019)
        df_main = cleanse_data(df_main,stk)
        df_main = prepare_data(df_main)
        df_main = merge_nse_nse_all(df_main)
        #df_main = merge_nse_quandl(df_main)
        if stk!='NIFTY_50': 
            df_main=merge_nse_stk(df_main,'NIFTY_50',2003,2019)

        
        #out=df_main[['Symbol',col_predict]].copy()
        #out['y']=out[col_predict].shift(-F_DAYS)
        
        
        #mod = GradientBoostingRegressor(n_estimators=96)
        #mod = RandomForestRegressor(n_estimators=200)
        
        out = predictor_AI(df_main,F_DAYS,col_predict,BATCH_SIZE,EPOCHS)
        out['Symbol']=stk+"_"+str(EPOCHS)+"_"+str(BATCH_SIZE)
        out_l.append(out)
        #out.dropna(axis=0).to_csv(stk+'_NSEAnalysisv3.csv')
    out = pd.concat(out_l,axis=0)
    out.to_csv('stocks_predicted.csv')
    return out


if __name__=='__main__':
    out = main()
    #out.dropna(axis=0)[['Close','y','pred']].plot()

