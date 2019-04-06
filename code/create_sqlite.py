import pandas as pd
from stock_lib import read_stk_data, read_yaml,update_stk_sql

# Stock list to load
stk_list = read_yaml('../configs/stocks.yml')
stk_list.append('NIFTY_50')
# Database file with historical data
db_file = "../data/stock_data.db"

## This is only for the first time update of the file from hdf5
# for stk in stk_list:
#     stk = stk.replace('','')
#     print("Processing :"+stk)
#     df = read_stk_data(stk,2003,2019)
#     df = df.fillna(method='ffill').fillna(method = 'bfill')
#     df.to_sql(stk,conn,if_exists='replace')

# conn.close()




