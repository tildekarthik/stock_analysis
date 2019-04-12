import pandas as pd
from stock_lib import read_yaml,read_stk_data_sql,momentum_back_test



db_name = '../data/stock_data.db'
stk_list_f = '../configs/stocks.yml' 
index_list_f = '../configs/indexes.yml'

# Model parameters
bt_days = 750
sw_length = 5
lw_length = 9


stk_list = read_yaml(stk_list_f)

out_l = []

for stk in stk_list:
    # Read the stk data
    l=[]
    try:
        df = read_stk_data_sql(stk,db_name)
        l = momentum_back_test(df,bt_days,sw_length,lw_length)
        l.insert(0,stk)
        out_l.append(l)
    except:
        print("Failed:"+stk)



df_csv = pd.DataFrame(out_l,columns=['Stock','Average_Price','Profit','Signals','sw','lw','btdays'])
df_csv['Profit_Pct']=df_csv['Profit']/df_csv['Average_Price']
df_csv.sort_values(by='Profit_Pct',ascending=False).to_csv('mom_output.csv')
