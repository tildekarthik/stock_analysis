import pandas as pd
from stock_lib import read_stk_data, read_yaml,update_stk_sql
# from datetime import timedelta
#import sqlite3 as lite
# from nsepy import get_history

stk_list = read_yaml('../configs/stocks.yml')
stk_list.append('NIFTY_50')

db_file = "../data/stock_data.db"
# conn = lite.connect(db_file)
# cur = conn.cursor()

## This is only for the first time update of the file from hdf5
# for stk in stk_list:
#     stk = stk.replace('','')
#     print("Processing :"+stk)
#     df = read_stk_data(stk,2003,2019)
#     df = df.fillna(method='ffill').fillna(method='bfill')
#     df.to_sql(stk,conn,if_exists='replace')

# conn.close()

# def update_stk_sql(stk):
#     df = pd.read_sql_query('select * from '+stk,conn, index_col='Date')
#     st_date = pd.to_datetime(df.index[-1])+timedelta(days=1)
#     end_date = pd.to_datetime("today")
#     if stk!='NIFTY_50':
#         df_add = get_history(symbol=stk, start=st_date, end=end_date)
#     else:
#         df_add = get_history(symbol=stk, start=st_date, end=end_date,index=True)
#     df = pd.concat([df,df_add], axis=0)
#     df = df.fillna(method='ffill').fillna('bfill')
#     return df





for stk in stk_list:
    stk=stk.replace('NSE/','')
    update_stk_sql(stk,db_file)
