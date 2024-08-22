
import pandas as pd
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import math
import akshare as ak
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import PercentFormatter
# 1.Get stocks

#Import Symbols
cs300 = pd.read_excel('CS300.xlsx',dtype='string')
csi1000 = pd.read_excel('CSI1000.xlsx',sheet_name=None,dtype='string')
years = [2015,2016,2017,2018,2019,2020,2021,2022]
cs300_years = {}
for year in years:
    cs300_years[year] = cs300[cs300['year'] == f'{year}'].reset_index(drop=True)

#Establish Every Year's Symbol
universe = {}
for year in years:
    cs300_on_year = cs300_years[year]
    csi1000_on_year = csi1000[f'{year}']

    cs300_on_year = cs300_on_year[['证券代码']].copy()
    cs300_on_year.rename(columns={'证券代码': 'symbol'}, inplace=True)

    csi1000_on_year = csi1000_on_year[['证券代码']].copy()
    csi1000_on_year.rename(columns={'证券代码': 'symbol'}, inplace=True)

    combined = pd.concat([cs300_on_year,csi1000_on_year],ignore_index=True).drop_duplicates().reset_index(drop=True)
    universe[year] = combined

#Get All year's Code
all_code_all_years = pd.DataFrame()
for year in years:
    all_code_all_years = pd.concat([all_code_all_years,universe[year]],ignore_index=True).drop_duplicates().reset_index(drop=True)


date = ak.tool_trade_date_hist_sina()
date['trade_date'] = pd.to_datetime(date['trade_date'])
date = date[(date['trade_date']>='2014-07-01') & (date['trade_date']< '2022-07-01')].reset_index(drop=True)


#Get Stocks' info 
all_stocks_insample= pd.DataFrame()
for i in range(len(all_code_all_years)):
    print(all_code_all_years.iloc[i]['symbol'])
    stockprice = ak.stock_zh_a_hist(symbol=all_code_all_years.iloc[i]['symbol'], period="daily", start_date="20140701", end_date='20220701', adjust="hfq")
    stockinfo = ak.stock_a_indicator_lg(symbol=all_code_all_years.iloc[i]['symbol'])
    #get price and fundatmental 
        

    stockprice = stockprice[['日期','股票代码','开盘','收盘','最高','最低','成交量','换手率']]
    stockprice.columns = ['trade_date','symbol','open','close','high','low','volume','turnover']
    stockinfo = stockinfo[['trade_date','pb','total_mv']]
    stockinfo['symbol'] = all_code_all_years.iloc[i]['symbol']
    stock = stockprice.merge(stockinfo,on=['trade_date','symbol'],how='left')
        #merge to one dataframe
    if not stockprice.empty:
        stock['trade_date'] = pd.to_datetime(stock['trade_date'])
        date_position = date['trade_date'].tolist().index(stock.loc[0,'trade_date'])
        date_to_merge = date['trade_date'].iloc[date_position:]
        stock = pd.merge(stock,date_to_merge,on='trade_date',how = 'outer')
        stock[['open','close','pb','total_mv','high','low','volume','turnover']]= stock[['open','close','pb','total_mv','high','low','volume','turnover']].interpolate(method='linear')
        #fill the date , interpolate 
            
        df = ak.stock_individual_info_em(symbol=all_code_all_years.iloc[i]['symbol'])
        ondate_delay_30= pd.to_datetime(df.loc[7,'value'],format='%Y%m%d') +  datetime.timedelta(days = 30)
        stock = stock.drop(stock[stock['trade_date'] < ondate_delay_30].index)
        stock['symbol'] = stock['symbol'].bfill()
        #drop the data that before listed 30 days 

        print(f'{i}:code:{all_code_all_years.iloc[i]['symbol']} success!')
        all_stocks_insample = pd.concat([all_stocks_insample,stock],ignore_index=True)
        
all_stocks_insample.to_csv(f'all_stocks_insample.csv')
print("save success!")







# 2. Calculate Alpha and Trade 

#Import Data
origin_stocks = pd.read_csv('all_stocks_insample.csv')
origin_stocks = origin_stocks.dropna(subset=['symbol']).reset_index(drop=True)
origin_date = origin_stocks[['trade_date']].drop_duplicates().reset_index(drop=True)
origin_stockcode = origin_stocks[['symbol']].drop_duplicates().reset_index(drop=True)


stocks = origin_stocks[['trade_date',  'symbol', 'open','close',  'pb',  'total_mv','high','low','volume','turnover']] #All history info
close = origin_stocks.pivot_table(index='trade_date',columns='symbol',values='close')#per stock per date's close

daily_return = close.pct_change(1)
delta_close = close.diff(1)
pre_close = close.shift(1)

date = origin_date['trade_date']#all date
stockcode = origin_stockcode['symbol']#all code

origin_date['Daily_Pnl'] = 0
Pnl = origin_date.pivot_table(index='trade_date',values='Daily_Pnl') #per date Pnl


GMV = 10000 #GMV
TimeWindow1 = 30 #Timewindow in raw alpha1
TimeWindow2 = 20 #Timewindow in raw alpha2
# 2.1 Calculate Raw_Alpha1

print('Start Calculating Raw_Alpha!')
stocks_with_raw_alpha=pd.DataFrame()
for i in range (len(stockcode)):
    #Get specific stock's history info from symbol
    stock_df_name = f"stock_{stockcode[i]}"
    df = stocks[stocks['symbol'] == stockcode[i]]
    df = df.reset_index(drop = True)

    #Calculate daily return / N days return / |daily return|
    df['daily return'] = df['close'].pct_change(periods = 1)
    df['abs daily return'] = abs(df['daily return'])
    df[f'{TimeWindow1} days return'] = df['close'].pct_change(periods = TimeWindow1)
    df['vol20'] = df['daily return'].shift(1).rolling(window=20,min_periods=20).std()

    #Calculate RS{N}
    df[f'RS{TimeWindow1}'] = 1 /  (df['daily return'].rolling(window = TimeWindow1,min_periods = TimeWindow1)
                                  .mean() + df['abs daily return'].rolling(window = TimeWindow1,min_periods = TimeWindow1).max())
    
    #Calculate WRD
    df['win or lose'] = abs(df['daily return'])/df['daily return']
    df['win or lose'] = df['win or lose'].fillna(0)
    df['win or lose'] = df['win or lose'].astype(int)
    df['WRD'] = abs ((df['win or lose'].rolling(window = TimeWindow1,min_periods = TimeWindow1).sum())/TimeWindow1)

    #Calculate TS
    df[f'Turn{TimeWindow1}'] = df['turnover'].rolling(window = TimeWindow1,min_periods = TimeWindow1).mean()
    df[f'STR{TimeWindow1}'] = df['turnover'].rolling(window = TimeWindow1,min_periods = TimeWindow1).std()
    df[f'TS{TimeWindow1}'] = 1/np.exp(df[f'Turn{TimeWindow1}'] + df[f'STR{TimeWindow1}'])
    

    #Calculate Raw_alpha
    df['raw alpha'] = df[f'{TimeWindow1} days return'] * df['WRD'] * df[f'RS{TimeWindow1}'] * df[f'TS{TimeWindow1}'] 
    if(i % 100 == 0 and i > 0 ):
        print(f"code number from {i-100} to {i}'s Raw_Alpha got!")
    stocks_with_raw_alpha = pd.concat([stocks_with_raw_alpha,df],ignore_index=True)
#Set Raw_alpha table
Raw_alpha1 = stocks_with_raw_alpha.pivot_table(index='trade_date', columns='symbol', values='raw alpha').shift(1)
#Set vol20 table
vol20 = stocks_with_raw_alpha.pivot_table(index='trade_date',columns='symbol',values='vol20')
print('All Stocks Raw_Alpha1 Got！')

#Get every year's stockcode , prepared to drop
cs300 = pd.read_excel('CS300.xlsx',dtype='string')
csi1000 = pd.read_excel('CSI1000.xlsx',sheet_name=None,dtype='string')
years = [2015,2016,2017,2018,2019,2020,2021,2022]
cs300_years = {}
for year in years:
    cs300_years[year] = cs300[cs300['year'] == f'{year}'].reset_index(drop=True)

universe = {}
for year in years:
    cs300_on_year = cs300_years[year]
    csi1000_on_year = csi1000[f'{year}']

    cs300_on_year = cs300_on_year[['证券代码']].copy()
    cs300_on_year.rename(columns={'证券代码': 'symbol'}, inplace=True)

    csi1000_on_year = csi1000_on_year[['证券代码']].copy()
    csi1000_on_year.rename(columns={'证券代码': 'symbol'}, inplace=True)

    combined = pd.concat([cs300_on_year,csi1000_on_year],ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined = combined[combined['symbol']!=600734.0]
    combined = combined[combined['symbol']!=600595.0]
    combined = combined[combined['symbol']!=600069.0]
    universe[year] = combined



# 2.2 Calculate Raw_Alpha2
#Calculate 3-factoer in fama-french
TF = pd.DataFrame(columns=['trade_date', 'R_market', 'SMB', 'HML'])
for i in range(len(date)):
    ondate = date[i]
    data_on_date = stocks_with_raw_alpha[stocks_with_raw_alpha['trade_date'] == ondate].reset_index(drop = True)
    ondate_time = pd.to_datetime(ondate).year
    #The Shanghai and Shenzhen 300 Index combined with the CSI 1000 Index from that year.
    if(ondate_time>2014):
        codes_on_year = universe[year]
        codes_to_keep = codes_on_year['symbol'].unique()
        codes_to_keep = codes_to_keep.astype(int)
        data_on_date = data_on_date[data_on_date['symbol'].isin(codes_to_keep)]
    data_on_date['BM'] = 1 / data_on_date['pb']
    sum_mv = data_on_date['total_mv'].sum()
    data_on_date['return_weight'] = data_on_date['daily return'] * data_on_date['total_mv'] / sum_mv
    MKT = (data_on_date['daily return'] * data_on_date['total_mv']).sum() / data_on_date['total_mv'].sum()
    
    # Group by BM/MV 
    data_on_date['BM_group'] = pd.qcut(data_on_date['BM'], q=[0, 0.3, 0.7, 1.0], labels=['L', 'M', 'H'])
    data_on_date['mv_group'] = pd.qcut(data_on_date['total_mv'], q=2, labels=['S', 'B'])
    data_on_date['group'] = data_on_date['BM_group'].astype(str) + '-' + data_on_date['mv_group'].astype(str)
    
    grouped = data_on_date.groupby('group')

    LS_group_data = grouped.get_group('L-S')
    LB_group_data = grouped.get_group('L-B')
    MS_group_data = grouped.get_group('M-S')
    MB_group_data = grouped.get_group('M-B')
    HS_group_data = grouped.get_group('H-S')
    HB_group_data = grouped.get_group('H-B')


    LS = (LS_group_data['daily return'] * LS_group_data['total_mv']).sum() / LS_group_data['total_mv'].sum()
    LB = (LB_group_data['daily return'] * LB_group_data['total_mv']).sum() / LB_group_data['total_mv'].sum()
    MS = (MS_group_data['daily return'] * MS_group_data['total_mv']).sum() / MS_group_data['total_mv'].sum()
    MB = (MB_group_data['daily return'] * MB_group_data['total_mv']).sum() / MB_group_data['total_mv'].sum()
    HS = (HS_group_data['daily return'] * HS_group_data['total_mv']).sum() / HS_group_data['total_mv'].sum()
    HB = (HB_group_data['daily return'] * HB_group_data['total_mv']).sum() / HB_group_data['total_mv'].sum()

    SMB = (LS + MS + HS) / 3  - (LB + MB + HB) / 3
    HML = (HS + HB) / 2 - (LS + LB ) /2
    TF = pd.concat([TF, pd.DataFrame([[ondate, MKT, SMB, HML]], columns=['trade_date', 'R_market', 'SMB', 'HML'])], ignore_index=True)


#Linear Regression Model to calculate IV20
for i in range(len(stockcode)):
    symbol = stockcode[i]
    stock_on_symbol = stocks[stocks['symbol']==symbol]
    stock_on_symbol = stock_on_symbol.reset_index()
    stock_on_symbol.loc[:, 'daily_return'] = stock_on_symbol['close'].pct_change(1)
    stock_to_regression = pd.merge(stock_on_symbol[['trade_date','daily_return']],TF,on='trade_date',how='inner')
    stock_to_regression.to_csv('test.csv')
    stock_to_regression = stock_to_regression.dropna().reset_index(drop=True) 
    for j in range (0,len(stock_to_regression)-TimeWindow2):
        window_for_regression = stock_to_regression.loc[j:j+TimeWindow2-1]
        window_for_regression = window_for_regression.reset_index(drop=True)
        X = window_for_regression[['R_market', 'SMB', 'HML']]
        Y = window_for_regression['daily_return']
        model = LinearRegression()
        model.fit(X, Y)

        coefficients = model.coef_
        intercept = model.intercept_

        window_for_regression['R_hat'] = window_for_regression['R_market'] * coefficients[0] + window_for_regression['SMB'] * coefficients[1] + window_for_regression['HML'] * coefficients[2] + intercept
        window_for_regression['Residual'] = window_for_regression['daily_return'] - window_for_regression['R_hat']
        alpha1 = window_for_regression['Residual'].std()
        stock_to_regression.loc[j+TimeWindow2,symbol] = alpha1
 
    IV20_ = pd.merge(IV20_,stock_to_regression[['trade_date',symbol]],on='trade_date',how='left')



IV20_ = IV20_.set_index('trade_date',drop=True)
IV20_.columns = IV20_.columns.astype(float)
IV20_ = IV20_.reindex(sorted(IV20_.columns), axis=1)
Raw_alpha2 = IV20_ / vol20

#The Shanghai and Shenzhen 300 Index combined with the CSI 1000 Index from that year.
Raw_alpha1.index = pd.to_datetime(Raw_alpha1.index)
Raw_alpha2.index = pd.to_datetime(Raw_alpha2.index)
for year in years:
    codes_on_year = universe[year]
    codes_to_keep = codes_on_year['symbol'].unique()
    codes_to_keep = codes_to_keep.astype(float)
    print(codes_to_keep)

    start_date = pd.to_datetime(str(year), format='%Y')
    end_date = pd.to_datetime(str(year+1), format='%Y')
    mask1 = (Raw_alpha1.index >= start_date) & (Raw_alpha1.index <= end_date)
    mask2 = (Raw_alpha2.index >= start_date) & (Raw_alpha2.index <= end_date)

    Raw_alpha1.loc[mask1, ~Raw_alpha1.columns.isin(codes_to_keep)] = np.nan
    Raw_alpha2.loc[mask2, ~Raw_alpha1.columns.isin(codes_to_keep)] = np.nan

daily_return.index = pd.to_datetime(daily_return.index)

# Get icir 
ic1= pd.DataFrame(index=Raw_alpha1.index, columns=['Correlation'])

# calculate ic
for idx in Raw_alpha1.index:
    row1 = Raw_alpha1.loc[idx]
    row2 = daily_return.loc[idx]
    
    # drop nan
    valid_idx = row1.dropna().index.intersection(row2.dropna().index)
    if len(valid_idx) > 1:  
        corr = row1[valid_idx].corr(row2[valid_idx])
    else:
        corr = np.nan  
    
    ic1.loc[idx, 'ic'] = corr
ic1['ir'] = ic1['ic'].rolling(window= TimeWindow2 , min_periods=1).std()
ic1['mean ic'] = ic1['ic'].rolling(window= TimeWindow2 , min_periods=1).mean()
ic1['icir'] = ic1['mean ic']/ic1['ir']

ic2= pd.DataFrame(index=Raw_alpha2.index, columns=['Correlation'])

# calculate ic
for idx in Raw_alpha2.index:
    row1 = Raw_alpha2.loc[idx]
    row2 = daily_return.loc[idx]
    
    # drpp nan
    valid_idx = row1.dropna().index.intersection(row2.dropna().index)
    if len(valid_idx) > 1:  
        corr = row1[valid_idx].corr(row2[valid_idx])
    else:
        corr = np.nan  
    
    ic2.loc[idx, 'ic'] = corr
ic2['ir'] = ic2['ic'].rolling(window=TimeWindow2 , min_periods=1).std()
ic2['mean ic'] = ic2['ic'].rolling(window=TimeWindow2 , min_periods=1).mean()
ic2['icir'] = ic2['mean ic']/ic2['ir']


#2.3. Refine Alpha1 & 2
print('Start Refining Raw_Alpha1')
Refined_alpha1 = {}
for i in range(TimeWindow1+1,len(date)):
    trade_date = date[i]

    alpha_on_date = Raw_alpha1.loc[trade_date] #Get raw_alpha(on date)
    not_none_count = alpha_on_date.count() #Get valid raw_alpha's count
    alpha_on_date = alpha_on_date.fillna(0) #fill NAN to 0 
    alpha_on_date = alpha_on_date.sort_values() #sort the raw_alpha
    
    #Get the short/long location from the count
    short_loc = ( not_none_count * 0.1 ).astype(int)
    long_loc = ( not_none_count * 0.9 ).astype(int)

    #Refine the raw alpha
    short_sum = abs(alpha_on_date.iloc[0:short_loc].sum())
    long_sum = abs(alpha_on_date.iloc[long_loc:].sum())
    
    alpha_on_date.iloc[0:short_loc] = alpha_on_date.iloc[0:short_loc] / short_sum / 2
    alpha_on_date.iloc[short_loc:long_loc] = 0
    alpha_on_date.iloc[long_loc:] =alpha_on_date.iloc[long_loc:] / long_sum / 2

    alpha_on_date = alpha_on_date.fillna(0)
    alpha_on_date = alpha_on_date.sort_index()

    Refined_alpha1[trade_date] = alpha_on_date
print('Refined!')


print('Start Refining Raw_Alpha2')
Refined_alpha2 = {}
for i in range(TimeWindow2+1,len(date)):
    trade_date = date[i]

    alpha_on_date = Raw_alpha2.loc[trade_date] #Get raw_alpha(on date)
    not_none_count = alpha_on_date.count() #Get valid raw_alpha's count
    alpha_on_date = alpha_on_date.sort_values() #sort the raw_alpha
    
    #Get the short/long location from the count
    short_loc = ( not_none_count * 0.1 ).astype(int)
    long_loc = ( not_none_count * 0.9 ).astype(int)

    #Refine the raw alpha
    short_sum = abs(alpha_on_date.iloc[0:short_loc].sum())
    long_sum = abs(alpha_on_date.iloc[long_loc:].sum())
    
    alpha_on_date.iloc[0:short_loc] = alpha_on_date.iloc[0:short_loc] / short_sum /  2
    alpha_on_date.iloc[short_loc:long_loc] = 0
    alpha_on_date.iloc[long_loc:] =alpha_on_date.iloc[long_loc:] / long_sum / -2
    
    alpha_on_date = alpha_on_date.fillna(0)
    alpha_on_date = alpha_on_date.sort_index()
    Refined_alpha2[trade_date] = alpha_on_date
print('Refined!')




#2.4.Trade and calculate PNL

Pnl['A1'] = np.nan
Pnl['A2'] = np.nan
def start_trade():
    #Set transaction cost
    Transaction_Fee = 0.0000341 #Fee
    bid_ask_spread = 0.0002 #bid ask spread 
    Slip_Point_Cost = 0.0002 #Slip Point
     
    Cost_for_dual = Transaction_Fee + bid_ask_spread + Slip_Point_Cost
    Stamp_Duty_for_Sale = 0.001

    Buy_Cost = Cost_for_dual
    Sell_Cost = Cost_for_dual + Stamp_Duty_for_Sale
    A1 = 0.5
    A2 = 0.5
    #for every trade date 
    for i in range(TimeWindow1+1 ,len(date)):

        trade_date = date[i]
        alpha1_on_date = Refined_alpha1[trade_date]
        alpha2_on_date = Refined_alpha2[trade_date]
        daily_return_on_date = daily_return.loc[trade_date]

        alpha1_pnl =   GMV * alpha1_on_date * daily_return_on_date * (1 - Buy_Cost - Sell_Cost)
        alpha2_pnl =   GMV * alpha2_on_date * daily_return_on_date * (1 - Buy_Cost - Sell_Cost)

        pnl_on_date = A1 * alpha1_pnl.sum() + A2 * alpha2_pnl.sum()
        Pnl.loc[trade_date,'Daily_Pnl'] = pnl_on_date   
        
        A1 = np.exp(20 * ic1.loc[trade_date]['mean ic'])
        A2 = np.exp(20 * ic2.loc[trade_date]['mean ic'])
        sum_A12 = A1 + A2
        A1 = A1 / sum_A12
        A2 = A2 / sum_A12
        
        Pnl.loc[trade_date,'A1'] =A1
        Pnl.loc[trade_date,'A2'] =A2
    Pnl['Sum_Pnl'] = Pnl['Daily_Pnl'].cumsum()#use cumsum to get cumulative pnl
    return Pnl


Pnl = start_trade()#Trade


#3 Calculate statistics

#3.1 Import Data
Timewindow = 30
Dates = 252
Pnl = Pnl.iloc[Timewindow:]
Pnl['volatility'] = (Pnl['Daily_Pnl']/10000 ).rolling(252).std()

#3.2Calculate Statistics
Statistics = pd.DataFrame(columns=['Annual Return','Volatility','Max Drawdown','SharpRatio','IR_Ratio','Win rate'])

Pnl['Max_Before'] = Pnl['Sum_Pnl'].expanding().max()
Statistics.loc[0,'Annual Return'] = (Pnl.iloc[-1]['Sum_Pnl'] / 10000 )  * (Dates/len(Pnl)) 
Statistics.loc[0,'Volatility'] = (Pnl['Daily_Pnl']/10000).std() * np.sqrt(252)
Statistics.loc[0,'Max Drawdown'] =- ((Pnl['Sum_Pnl'] - Pnl['Max_Before']) / (Pnl['Max_Before']+ 10000)).min() 
Statistics.loc[0,'Win rate'] = (Pnl['Daily_Pnl']>0).sum()/len(Pnl)

Statistics['SharpRatio'] = (Statistics['Annual Return'] - 0.03 ) / Statistics['Volatility']
Statistics['IR_Ratio'] = Statistics['Annual Return'] / Statistics['Volatility']
Statistics = Statistics.style.format({
    'Annual Return': '{:.2%}',
    'Volatility': '{:.2%}',
    'IR_Ratio':'{:.2f}',
    'SharpRatio':'{:.2f}',
    'Max Drawdown': '{:.2%}',
    'Win rate': '{:.2%}',
})
from IPython.display import display
display(Statistics)




#3.3Show PNL
plt.figure(figsize=(10, 6))
Pnl.index = pd.to_datetime(Pnl['trade_date'])
plt.plot(Pnl.index, Pnl['Sum_Pnl']/10000,color='black', label='Refined LST-SM Alpha')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.title("PNL on Origin-Refined Alpha:")
plt.xlabel("Date")
plt.ylabel("The percentage of PNL relative to the GMV")

plt.ylim
plt.grid(True)
plt.legend(loc='upper left')
plt.xticks(rotation=45)

plt.show()
Pnl.to_csv('Pnl.csv')




