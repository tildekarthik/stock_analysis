
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:07:04 2018

@author: karthik
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from stock_lib import *



def main():
    # Run 10 day with RFR
    F_DAYS = 10
    estimators = 250
    stk_list = ['LT','TATAMOTORS']
    col_predict = 'Close'


    # Run 1  and 5 day with GBR - manually switch mod functions
    # F_DAYS = 5
    # estimators = 100
    # stk_list = ['HDFCBANK','AUROPHARMA'] 
    # col_predict = 'Close'    

    # F_DAYS = 1
    # estimators = 100
    # stk_list = ['TCS','SUNPHARMA','HINDUNILVR']
    # # col_predict = 'close-open' 
    # col_predict = 'diff_1Close'

    out_l = []
    for stk in stk_list:
        print("Processing:" + stk)
        df_main = read_stk_data(stk, 2014, 2019)
        df_main = cleanse_data(df_main, stk)
        df_main = prepare_data(df_main)
        df_main = merge_nse_nse_all(df_main)
        # df_main = merge_nse_quandl(df_main)
        if stk != 'NIFTY_50':
            df_main = merge_nse_stk(df_main, 'NIFTY_50', 2014, 2019)
        # prepare the output format
        out = df_main[['Symbol', col_predict]].copy()
        out['y'] = out[col_predict].shift(-F_DAYS)
        # mod = GradientBoostingRegressor(n_estimators=estimators)
        mod = RandomForestRegressor(n_estimators=estimators)
        out = predictor(df_main, F_DAYS, col_predict, mod)
        out['Symbol'] = stk
        out_l.append(out.dropna(axis=0))
    out = pd.concat(out_l, axis=0)
    out.to_csv('stocks_predicted.csv')
    print(out)
    return out


if __name__ == '__main__':
    out = main()
    # out.dropna(axis=0)[['Close','y','pred']].plot()
