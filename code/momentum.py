import pandas as pd


def read_stk_data(stk,start_year,end_year):
    l=[]
    for i in range(start_year,end_year+1):
        l.append(pd.read_hdf('nsepy_'+str(i)+'.hdf5',key=stk.replace(' ','_')))
    df_read=pd.concat(l,axis=0)
    return df_read

def cleanse_data(df_m,stk):
    if 'Trades' in df_m.columns:
        df_m.drop('Trades',axis=1,inplace=True)
    df_m['Symbol']=stk
    return df_m


def mom_back_test(stk, st_year, end_year,   sw_days, lw_days):
    
    pass
    

df = read_stk_data('HINDUNILVR',2015,2019)

LW_DAYS = 21
SW_DAYS = 5

df['LW_by_SW']=df['Close'].ewm(LW_DAYS).mean()/df['Close'].ewm(SW_DAYS).mean()
df['LW_by_SW_yesterday']=df['LW_by_SW'].shift(1)
df['Long_Short']=0

df.loc[(df['LW_by_SW']>1) & (df['LW_by_SW_yesterday']<1),'Long_Short']=1
df.loc[(df['LW_by_SW']<1) & (df['LW_by_SW_yesterday']>1),'Long_Short']=-1

df_switch = df.loc[df['Long_Short']!=0,['Close','Long_Short']]
df_switch['exit_price']=df_switch['Close'].shift(-1)
df_switch['exit_date'] = df_switch.index
df_switch['exit_date'] = df_switch['exit_date'].shift(-1)
df_switch = df_switch[df_switch['Long_Short']==1]
df_switch.to_csv('out.csv')
