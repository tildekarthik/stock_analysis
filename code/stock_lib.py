#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:07:04 2018

@author: karthik

#V2 - ADDED THE DIFF COLUMN TO MAKE IT BEHAVE LIKE ARIMA! in prepare _data
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt
from yaml import load

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


import sqlite3 as lite
from datetime import timedelta
from nsepy import get_history

def read_yaml(file_w_path):
    # import yaml file from directory and return the list   
    f_i = open(file_w_path)
    out = load(f_i)
    f_i.close()
    return out


def update_stk_sql(stk,db_file):
    conn = lite.connect(db_file)
    cur = conn.cursor()
    
    try:
        df = pd.read_sql_query('select * from '+stk,conn, index_col='Date')
        st_date = pd.to_datetime(df.index[-1])+timedelta(days=1)
        end_date = pd.to_datetime("today")
        if stk!='NIFTY_50':
            df_add = get_history(symbol=stk, start=st_date, end=end_date)
        else:
            df_add = get_history(symbol=stk, start=st_date, end=end_date,index=True)
        df = pd.concat([df,df_add], axis=0)
        df = df.fillna(method='ffill').fillna('bfill')
        print("Processing : "+stk+" : "+str(df.index[-1]))
        df.to_sql(stk,conn,if_exists='replace')
    except:
        print("Failed : "+stk)
    
    cur.close()
    conn.close()
    return df




#from sklearn.ensemble import RandomForestRegressor
def stock_close_NSE(co_name_list,st_year,end_year):
    '''
    returns close prices of all the stocks in a pandas dataframe for the period specified
    '''
    l=[]
    for yr in range(st_year,end_year+1):    
        temp=[]
        for stk in co_name_list:
            print (yr,stk)
            stk=stk.replace('','')
            st_df = pd.read_hdf('nsepy_'+str(yr)+'.hdf5',key=stk.replace(' ','_')).add_suffix(stk)['Close'+stk]
            st_df.index = pd.to_datetime(st_df.index)
            temp.append(st_df)
        l.append(pd.concat(temp,axis=1))
    df_out=pd.concat(l,axis=0)
    df_out = df_out.fillna(method='bfill').dropna(axis=1)
    df_out.to_csv('all_stock_NSE.csv')
    return df_out

# Buggy
def stock_close_NSE_sql(stk_db_name, stk_list):
    concat_list = []
    for stk in stk_list:
        stk=stk.replace('','')
        df = read_stk_data_sql(stk,stk_db_name)
        concat_list.append(df.loc[:,['Close']].rename(columns={'Close':'Close'+stk}))
    return pd.concat(concat_list,axis=1,verify_integrity=False)
        


def read_stk_data(stk,start_year,end_year):
    l=[]
    for i in range(start_year,end_year+1):
        l.append(pd.read_hdf('nsepy_'+str(i)+'.hdf5',key=stk.replace(' ','_')))
    df_read=pd.concat(l,axis=0)
    return df_read

def read_stk_data_sql(stk,db_name):
    conn = lite.connect(db_name)
    cur = conn.cursor()
    df = pd.read_sql_query('select * from '+stk,conn, index_col='Date', parse_dates=True)
    conn.close()
    return df

def cleanse_data(df_m,stk):
    if 'Trades' in df_m.columns:
        df_m.drop('Trades',axis=1,inplace=True)
    df_m['Symbol']=stk
    return df_m


def prepare_data(df):
    shifts=[1,2,3,5,6,7,8,9,10,20,40,60,100,240]
    df['close-open']=df['Close']-df['Open']
    df['high-close']=df['High']-df['Close']
    df['close-low']=df['Close']-df['Low']
    df['open-low']=df['Open']-df['Low']
    df['high-open']=df['High']-df['Open']
    col_shift = ['Open','Close','close-open']
    for i in shifts:
        df = pd.concat([df, df[col_shift].shift(i).add_prefix('prev_'+str(i))],axis=1)
        # below has been added in v2 - needs to be back tested 
        df = pd.concat([df, df[col_shift].diff(i).add_prefix('diff_'+str(i))],axis=1)
        df = pd.concat([df, df[col_shift].ewm(i).mean().add_prefix('rolling_'+str(i))],axis=1)
    return(df)

def merge_nse_quandl(df1):
    df1.index = pd.to_datetime(df1.index)
    df_quandl = pd.read_csv('all_mega_stocks.csv', index_col='Date', parse_dates=True)
    df2=pd.concat([df_quandl,df1],axis=1)
    df2.dropna(axis=0,inplace=True)
    return df2


def merge_nse_nse_all(df1):
    df1.index = pd.to_datetime(df1.index)
    df_nse = pd.read_csv('all_stock_NSE.csv', index_col=0, parse_dates=True)
    try:
        df2=pd.concat([df_nse.dropna(axis=1),df1],axis=1)
    except:
        df2 = df1.join(df_nse.dropna(axis=1))
    df2.fillna(method='bfill',inplace=True)
    df2.fillna(method='ffill',inplace=True)
    df2.dropna(axis=0,inplace=True)
    return df2
    

def merge_nse_stk(df_1,stk_to_add,st_year,end_year):
    df_1.index = pd.to_datetime(df_1.index)
    
    df_2 = read_stk_data(stk_to_add,st_year,end_year)
    df_2 = cleanse_data(df_2,stk_to_add)
    df_2 = prepare_data(df_2)
    df_2.index = pd.to_datetime(df_2.index)
    df_2.drop('Symbol',inplace=True,axis=1)
    df_2=df_2.add_prefix(stk_to_add+'_')
    try:
        df_out=pd.concat([df_1.drop_duplicates(),df_2.drop_duplicates()],axis=1)
    except:
        df_out=df_1.drop_duplicates().join(df_2.drop_duplicates())
    df_out.dropna(axis=0,inplace=True)
    return df_out   

        
def X_y(df_X,col_predict,future_days, sample_length=0):
    mf = df_X._get_numeric_data().copy()
    X_test=mf.tail(1)
    mf['y']=mf[col_predict].shift(-future_days)
    mf.dropna(axis=0,inplace=True)
    y_train=mf.loc[:,['y']][-sample_length:]
    X_train=mf.drop(['y'],axis=1)[-sample_length:]
    return X_train,y_train,X_test


def train_pred(X_tr,y_tr,X_te,fn):
    fn.fit(X_tr,y_tr.values.ravel())
    return fn.predict(X_te)


def train_pred_AI(X_train_scaled,y_train_scaled,X_test_scaled,BATCH_SIZE,EPOCHS):
    input_dimension = X_train_scaled.shape[1]
    
    regression = Sequential()
    # Adding the input layer and the first hidden layer
    regression.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dimension))
    
#    # Adding the second hidden layer
#    regression.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the second hidden layer
    #regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    regression.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    
    # Compiling the ANN
    regression.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
    
    # Fitting the ANN to the Training set
    history = regression.fit(X_train_scaled, y_train_scaled, batch_size = BATCH_SIZE, epochs = EPOCHS,verbose=0)
    #plt.plot(history.history['mean_squared_error'])
    #plt.show()
    #inverse transform to get the actual values
    pred_test_scaled=regression.predict(X_test_scaled)

    return pred_test_scaled



def back_test(back_test_days,df_master,F_DAYS,col_predict,fn,out):
    for j in range(-back_test_days,0):
        print('Back test day: '+str(j))
        df=df_master[:j].copy()
        X_train,y_train,X_test = X_y(df,col_predict,F_DAYS,300)
        pred = train_pred(X_train,y_train,X_test,fn)
        ix=X_test.index[0]
        out.loc[ix,'pred']=pred
        #print(out.loc[ix,:])
    return out

def predictor(df_master,F_DAYS,col_predict,fn):
    X_train,y_train,X_test = X_y(df_master,col_predict,F_DAYS,300)
    pred = train_pred(X_train,y_train,X_test,fn)
    X_test['pred']=pred[0]
    return X_test[['Close','pred']]

def scale_X_y(X_train,y_train,X_test):
    scX = StandardScaler()
    X_train_scaled = scX.fit_transform(X_train)
    X_test_scaled = scX.transform(X_test)
    scy = StandardScaler()
    y_train_scaled = scy.fit_transform(y_train)
    return scX,scy,X_train_scaled,y_train_scaled,X_test_scaled

def back_test_AI_NSE(back_test_days,df_master,F_DAYS,col_predict,out,BATCH_SIZE,EPOCHS):
    #build_model and train, predict, scale back and pass result
    for j in range(-back_test_days,0):
        print('Back test day: '+str(j))
        df=df_master[:j].copy()
        X_train,y_train,X_test = X_y(df,'Close',F_DAYS,0)
        scX,scy,X_train_scaled,y_train_scaled,X_test_scaled = scale_X_y(X_train,y_train,X_test)        
        pred = train_pred_AI(X_train_scaled,y_train_scaled,X_test_scaled,BATCH_SIZE,EPOCHS)
        ix=X_test.index[0]
        out.loc[ix,'pred']=scy.inverse_transform(pred)[0,0]
    return out

def predictor_AI(df_master,F_DAYS,col_predict,BATCH_SIZE,EPOCHS):
    X_train,y_train,X_test = X_y(df_master,'Close',F_DAYS,0)
    scX,scy,X_train_scaled,y_train_scaled,X_test_scaled = scale_X_y(X_train,y_train,X_test)        
    pred = train_pred_AI(X_train_scaled,y_train_scaled,X_test_scaled,BATCH_SIZE,EPOCHS)
    X_test['pred']=scy.inverse_transform(pred)[0,0]
    return X_test[['Close','pred']]



def train_pred_RNN(X_train_scaled,y_train_scaled,X_test_scaled,BATCH_SIZE,EPOCHS):
    
    input_dimension = X_train_scaled.shape[1]
    
    X_train = []
    y_train = []
    for i in range(30, len(df_train)):
        X_train.append(X_train_scaled[i-30:i,:])
        y_train.append(y_train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], input_dimension))
    
    input_dimension = X_train_scaled.shape[1]
    
    regression = Sequential()
    # Adding the put layer and the first hidden layer
    regression.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dimension))
    
#    # Adding the second hidden layer
#    regression.add(Dense(units = 60, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the second hidden layer
    #regression.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    regression.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    
    # Compiling the ANN
    regression.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
    
    # Fitting the ANN to the Training set
    history = regression.fit(X_train_scaled, y_train_scaled, batch_size = BATCH_SIZE, epochs = EPOCHS,verbose=0)
    #plt.plot(history.history['mean_squared_error'])
    #plt.show()
    #inverse transform to get the actual values
    pred_test_scaled=regression.predict(X_test_scaled)

    return pred_test_scaled


# Candle pattern analyser
def inrange(pt,st,end):
    if (pt>st) & (pt<end):
        return True
    else:
        return False
        
def prepare_candle_params(df):
    # Set candle color and then the pt2 and pt3 params
    df['Green']=0
    df.loc[df['Close']>df['Open'],'Green']=1
    df['pt2']=df['Close']
    df['pt3']=df['Open']

    df.loc[df['Green']==0,'pt2']=df.loc[df['Green']==0,'Open']
    df.loc[df['Green']==0,'pt3']=df.loc[df['Green']==0,'Close']

    # Set wick params
    df['top_wick']= df['High']-df['pt2']
    df['body']=df['pt2']-df['pt3']
    df['bot_wick']=df['pt3']-df['Low']
    # Wick parameters
    df['tw_ratio'] = df['top_wick']/df['body']
    df['bw_ratio'] = df['bot_wick']/df['body']
    df['body_rel_size'] = df['body']/df['body'].rolling(20).mean()
    df['rel_vol'] = df['Volume']/df['Volume'].rolling(10).mean()
    # MA params
    df['ma_5'] = df['Close'].ewm(5).mean()
    df['ma_9'] = df['Close'].ewm(9).mean()
    df['ma_21'] = df['Close'].ewm(21).mean()
    # Trend
    df['diff_13'] = df['ma_5'].diff(13)
    df['diff_21'] = df['ma_5'].diff(21)

    # Volume
    df['Volume_MA'] = df['Volume']/df['Volume'].rolling(10).mean()

    # Collate last 3 candles
    today = dict(df.iloc[-1,:])
    yest = dict(df.iloc[-2,:])
    day_bef = dict(df.iloc[-3,:])
    return (today,yest,day_bef)



def candle_pattern_analyser(today,yest,day_bef):
    symbols = {}


    # write algos for checking each and fill the results
    #MARUBOZU
    symbols['marubozu'] = (today['tw_ratio']<0.05) & (today['bw_ratio']<0.05)
    symbols['doji']  = (today['body_rel_size']<0.25) & (today['tw_ratio']>2) & (today['bw_ratio']>2)
    symbols['hammer'] = (today['body_rel_size']>0.25) & (today['tw_ratio']<.05) & (today['bw_ratio']>2)

    #2 day
    symbols['engulfing_harami']=((today['pt2']>yest['pt2']) == (today['pt3']<yest['pt3'])) & (today['Green']!=yest['Green'])
    # piercing logic
    inversion = (today['Green']!=yest['Green'])
    top_in_range = inrange(yest['pt3'],yest['pt2'],today['pt2'])
    bot_in_range = inrange(yest['pt3'],yest['pt2'],today['pt3'])
    only_one = (top_in_range!=bot_in_range)
    symbols['piercing'] = inversion & only_one


    # 3 day
    symbols['star']= ((day_bef['pt3']-yest['pt2']>0) & (today['pt3']-yest['pt2']>0)) | ((yest['pt3']-day_bef['pt2']>0)&(yest['pt3']-today['pt2']>0))

    # Volume
    symbols['volume'] = today['Volume_MA']>.8

    return symbols




