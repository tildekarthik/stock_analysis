#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:07:04 2018

@author: karthik
"""
import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
from stock_lib import *


def main():
    F_DAYS = 5
    bt_days = 250
    # col_predict = 'close-open'
    # col_predict='diff_1Close'
    col_predict='Close'
    f_out_name = 'stocks' + str(F_DAYS) + '.csv'
    # Below code is trying to directly predict the diff
    out_l = []
#    stk_list=['COLPAL','RELIANCE']

    stk_list = ['NIFTY_50', 'ASIANPAINT', 'AUROPHARMA', 'COLPAL', 'CADILAHC', 'FINCABLES', 'FEDERALBNK', 'GLAXO',
                'GRASIM', 'HDFCBANK', 'ICICIBANK', 'INFY', 'ITC', 'KOTAKBANK', 'TATAMOTORS', 'RELIANCE',
                'HINDUNILVR', 'HINDALCO', 'LT', 'MARICO', 'SBIN', 'SUNPHARMA', 'TATACHEM',
                'TATASTEEL', 'TCS', 'TITAN']
    estimators = 100

    for stk in stk_list:
        print("Processing:" + stk)
        df_main = read_stk_data(stk, 2014, 2019)
        df_main = cleanse_data(df_main, stk)
        df_main = prepare_data(df_main)
        df_main = merge_nse_nse_all(df_main)
        # df_main = merge_nse_quandl(df_main)
        # if stk != 'NIFTY_50':
        #     df_main = merge_nse_stk(df_main, 'NIFTY_50', 2014, 2019)

        out = df_main[['Symbol', col_predict]].copy()
        out['y'] = out[col_predict].shift(-F_DAYS)

        mod = GradientBoostingRegressor(n_estimators=estimators)
        # mod = RandomForestRegressor(n_estimators=estimators)
        # mod = SVR(kernel='rbf', gamma=10000, C=10000)

        out = back_test(bt_days, df_main, F_DAYS, col_predict, mod, out)
        out_l.append(out.dropna(axis=0))
    out = pd.concat(out_l, axis=0)
    out.to_csv(f_out_name)
    return out


if __name__ == '__main__':
    out = main()
    # out.dropna(axis=0)[['Close','y','pred']].plot()
