# coding: utf-8
import pandas as pd
from pandas import DataFrame
from pandas import DataFrame
from pandas import concat
from Stock_Interface import read_finance_year
import datetime


# 更新股票基础信息 
def to_stock_list(pro):
    df = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,area,industry,list_date,fullname,market,exchange,list_status,is_hs')
    df = df.to_excel('../JN_DataWarehouse/stock_analysis/TuShare/stock_list.xls')
    

# 更新交易日历
def to_cal(pro):
    df = pro.trade_cal(exchange='', start_date='20010101', end_date='20290101', fields='exchange,cal_date,is_open,pretrade_date')
    df = df.to_excel('../JN_DataWarehouse/stock_analysis/TuShare/cal.xls')
    

# 更新上市公司基本信息
def to_stock_company(pro):
    df = pro.stock_company()
    df = df.to_excel('../JN_DataWarehouse/stock_analysis/TuShare/company.xls')
    

# 更新复权因子信息
def to_adjfactor(df):
    df = df.to_csv('../JN_DataWarehouse/stock_analysis/TuShare/adjfactor.csv')


# 读取复权因子信息
def read_adjfactor(start_date, end_date):
    df = pd.read_csv('../JN_DataWarehouse/stock_analysis/TuShare/adjfactor.csv', index_col=False)
    df_adjfactor = df.query(" trade_date >= "+start_date+" and trade_date <= "+end_date+"")[['ts_code','trade_date','adj_factor']]  # query 查询语句
    return df_adjfactor
    

# 读取日历
def read_cal():
    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/TuShare/cal.xls', index_col=False)
    return df


# 读取日历，按时间段
def read_cal_t(start_date, end_date):
    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/TuShare/cal.xls', index_col=False)
    df_cal_t = df.query("is_open == 1 and cal_date >= "+start_date+" and cal_date <= "+end_date+"")[['cal_date']]  # query 查询语句
    return df_cal_t
           
    
    
# 读取year
def read_year(year):
    df = pd.read_csv('../JN_DataWarehouse/stock_analysis/TuShare/'+year+'.csv', index_col=False)
    del df['Unnamed: 0']
    return df

# 更新 year
def to_year(df, year):
    df = df.to_csv('../JN_DataWarehouse/stock_analysis/TuShare/'+year+'.csv')

# 更新 year
def to_newyear(df):
    df = df.to_csv('../JN_DataWarehouse/stock_analysis/TuShare/new_year.csv')
    

    
# 生成20year的日数据按年合并
def build_daily(pro, year):
    
    start_date = year+'0101'
    end_date = year+'1231'
    
    df_cal_t = read_cal_t(start_date, end_date)
    count = int( df_cal_t.count() )
    #print(count)
    
    df_year = pro.daily(start_date='?', end_date='?')
    #print(df_year)
    for r in zip(df_cal_t['cal_date']):
        #print(r[0])
    
        df_daily = pro.daily(start_date=r[0], end_date=r[0])
    
        df_year = df_year.append(df_daily, ignore_index = True)
    
        #time.sleep(1)
    
        count = count - 1
        if count == 0:
            break
    
    to_year(df_year, year)
    print('ok!')
    
    

#读取XX公司日数据，按时间段
def build_stock_newyear():
    years = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
             '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020',
             '2021','2022']
    df = read_year('2000')
        
    for x in years:
        df1 = read_year(x)
        df = concat([df,df1],ignore_index=True).drop_duplicates() 
    
    #del df['Unnamed: 0']
    
    to_newyear(df)
    
    return df


# 生成20年前复权因子
def build_adjfactor(pro, year):
    
    start_date = year+'0101'
    end_date = year+'1231'
    
    num_start = int(year+'0000')
    num_end = int(year+'9999')
    
    df_cal_t = read_cal_t(start_date, end_date)
    count = int( df_cal_t.count() )
    #print(count)
    
    df_adjfactor = pd.read_csv('../JN_DataWarehouse/stock_analysis/TuShare/adjfactor.csv', index_col=False)
    del df_adjfactor['Unnamed: 0']
    df_adjfactor = df_adjfactor.loc[(df_adjfactor["trade_date"] < num_start) | (df_adjfactor["trade_date"] > num_end),['ts_code','trade_date','adj_factor']]
    
    #print(df_year)
    for r in zip(df_cal_t['cal_date']):
      
        df_day_adjfactor = pro.adj_factor(trade_date=r[0])
    
        df_adjfactor = df_adjfactor.append(df_day_adjfactor, ignore_index = True)
    
        #time.sleep(1)
    
        count = count - 1
        if count == 0:
            break
    
    to_adjfactor(df_adjfactor)
    print('ok!')
    

def build_stock_newyear_adjfactor():
    
    df_all_adjfactor = read_adjfactor('20010101','20250215')
    
    print(df_all_adjfactor.loc[df_all_adjfactor.ts_code == '000001.SZ',['ts_code','trade_date','adj_factor']].groupby(['ts_code','trade_date','adj_factor']).count().tail(5))
    
    today = datetime.date.today()
    today = today -  datetime.timedelta(2)
    countDate = today.strftime('%Y%m%d')
    countDate
    
    df_new_adjfactor = read_adjfactor(countDate,countDate)
    
    df_new_adjfactor = df_new_adjfactor.loc[df_new_adjfactor.ts_code.notnull(),['ts_code','adj_factor']]
    
    df_alldata = read_finance_year.read_newyear()
    
    df_newalldata = pd.merge(df_alldata, df_all_adjfactor, on=['trade_date','ts_code'])
    df_newalldata = pd.merge(df_newalldata, df_new_adjfactor, on=['ts_code'])
    df_newalldata['close1'] = (df_newalldata['close']*df_newalldata['adj_factor_x']/df_newalldata['adj_factor_y']).values.astype(float).round(3)
    
    print('\n')
    print(df_newalldata[(df_newalldata.ts_code == '601939.SH') & (df_newalldata.trade_date > 20200615)].tail(20))
    
    #保存最新结果
    to_newyear(df_newalldata)
    print('year!!')
    