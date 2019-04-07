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
    F_DAYS=10
    bt_days=30
    col_predict='Close'
    out_l = []
    stk_list = ['HDFCBANK','BANKNIFTY','NIFTY_50',
                 'DRREDDY','ASIANPAINT','CIPLA','RBLBANK',
                 'HDFC','UPL', 'HINDUNILVR', 'DIVISLAB',
                 'KOTAKBANK','TATACHEM', 'HAVELLS', 'TCS',
                 'CONCOR','SBIN', 'CADILAHC', 'ITC',
                 'SCHAEFFLER', 'BPCL', 'AMBUJACEM', 'TITAN',
                 'INDUSINDBK', 'TATAMOTORS', 'MINDTREE', 'INDIGO',
                 'BHARTIARTL', 'SUNPHARMA', 'PAGEIND', 'FEDERALBNK',
                 'AUROPHARMA','GAIL','HINDALCO','COALINDIA',
                 'GRASIM','LT']
    grid = {}
    for i in range(len(stk_list)):
        grid[i]=[stk_list[i],32,32]         
    
    
#    grid = {
##            1:['NIFTY_50',8,16],
##            2:['NIFTY_50',16,16],
##            3:['NIFTY_50',32,16],
##            4:['NIFTY_50',64,16],
##            5:['NIFTY_50',128,16],
##            6:['NIFTY_50',256,16],
##            7:['BANKNIFTY',16,32],
##            8:['BANKNIFTY',16,32],
##            9:['BANKNIFTY',32,32],
##            10:['BANKNIFTY',64,32]
##            11:['NIFTY_50',128,32],
##            12:['NIFTY_50',256,32]
##            1:['NIFTY_50',16,32],
##            2:['NIFTY_50',16,16],
##            3:['NIFTY_50',16,64]
#            }
#    grid={
#            1:['NIFTY_50',16,32],
#            2:['BANKNIFTY',16,32],
#            3:['LT',16,32],
#            4:['TCS',16,32],
#            5:['GRASIM',16,32]
#    stk_list=['NIFTY_50','BANKNIFTY','HINDUNILVR','TITAN','ASIANPAINT','HDFCBANK','GRASIM','GLAXO','AUROPHARMA','TITAN','ICICIBANK']
#            }
#    stk_list=['ASIANPAINT','AUROPHARMA','COLPAL','CADILAHC','FINCABLES','FEDERALBNK','GLAXO',
#                'GRASIM','HDFCBANK','ICICIBANK','INFY','ITC','KOTAKBANK','TATAMOTORS','RELIANCE',
#                'HINDUNILVR','HINDALCO','LT','MARICO','SBIN','SUNPHARMA','TATACHEM',
#                'TATASTEEL','TCS','TITAN']    

    for iter in grid.keys():
        print("Processing:"+str(iter))
        stk,EPOCHS,BATCH_SIZE=grid[iter]

        df_main = read_stk_data(stk,2003,2019)
        df_main = cleanse_data(df_main,stk)
        df_main = prepare_data(df_main)
        df_main = merge_nse_nse_all(df_main)
        #df_main = merge_nse_quandl(df_main)
        if stk!='NIFTY_50': 
            df_main=merge_nse_stk(df_main,'NIFTY_50',2003,2019)

        
        
        out=df_main[['Symbol',col_predict]].copy()
        out['y']=out[col_predict].shift(-F_DAYS)
        
        
        #mod = GradientBoostingRegressor(n_estimators=96)
        #mod = RandomForestRegressor(n_estimators=200)
        
        out=back_test_AI_NSE(bt_days,df_main,F_DAYS,col_predict,out,BATCH_SIZE,EPOCHS)
        out['Symbol']=out['Symbol']+"_"+str(EPOCHS)+"_"+str(BATCH_SIZE)
        out_l.append(out.dropna(axis=0))
        out.dropna(axis=0).to_csv('NSEAnalysisv3_Output_'+stk+'.csv')
    out = pd.concat(out_l,axis=0)
    out.to_csv('stocks.csv')
    return out


if __name__=='__main__':
    out = main()
    #out.dropna(axis=0)[['Close','y','pred']].plot()

