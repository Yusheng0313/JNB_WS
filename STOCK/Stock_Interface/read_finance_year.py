
# coding: utf-8
import pandas as pd
from pandas import DataFrame
from pandas import concat
from pandas import merge

years = {'2001','2002','2003','2004','2005','2006','2007','2008','2009','2010',
                 '2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'}

months = {'1','2','3','4'}
# In[ ]:


# 读取BPS每股净资产季度转年
def read_bps(stock_code):
    
    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/stockinfo/每股剩资产BPS.xlsx' ,index_col=False)
    df.pop('序号')
        
    for x in years:
        for y in months:
            if y != '4' :
                df.pop(x+'-'+y)
            else:
                df.rename(columns={x+'-'+y:x}, inplace = True)
    
    df.rename(columns={'股票代码':'ts_code'}, inplace = True)
    df.rename(columns={'股票简称':'name'}, inplace = True)
    
    df = df.query( "ts_code.str.contains('" + stock_code + "')")                
   
    return df

# 读取roe净资产收益率季度转年
def read_roe(stock_code):

    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/stockinfo/净资产收益率ROE.xlsx' ,index_col=False)
    df.pop('序号')

    for x in years:
        for y in months:
            if y != '4' :
                df.pop(x+'-'+y)
            else:
                df.rename(columns={x+'-'+y:x}, inplace = True)
    
    df.rename(columns={'股票代码':'ts_code'}, inplace = True)
    df.rename(columns={'股票简称':'name'}, inplace = True)
    
    df = df.query( "ts_code.str.contains('" + stock_code + "')")                
   
    return df
	
# 读取P/E市盈率季度转年
def read_pe(stock_code):

    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/stockinfo/市盈率PE.xlsx' ,index_col=False)
    df.pop('序号')

    for x in years:
        for y in months:
            if y != '4' :
                df.pop(x+'-'+y)
            else:
                df.rename(columns={x+'-'+y:x}, inplace = True)
    
    df.rename(columns={'股票代码':'ts_code'}, inplace = True)
    df.rename(columns={'股票简称':'name'}, inplace = True)
    
    df = df.query( "ts_code.str.contains('" + stock_code + "')")                
   
    return df

# 读取EPS每股收益
def read_eps(stock_code):

    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/stockinfo/每股收益EPS.xlsx' ,index_col=False)
    df.pop('序号')

    for x in years:
        for y in months:
            if y != '4' :
                df.pop(x+'-'+y)
            else:
                df.rename(columns={x+'-'+y:x}, inplace = True)
    
    df.rename(columns={'股票代码':'ts_code'}, inplace = True)
    df.rename(columns={'股票简称':'name'}, inplace = True)
    
    df = df.query( "ts_code.str.contains('" + stock_code + "')")
   
    return df


# 读取分红年度
def read_bonus(stock_code):
    
    df = pd.read_excel('../JN_DataWarehouse/stock_analysis/stockinfo/分红年度.xlsx' ,index_col=False)
    df.pop('序号')
        
    df.rename(columns={'股票代码':'ts_code'}, inplace = True)
    df.rename(columns={'股票简称':'name'}, inplace = True)
    #df.rename(columns={2020:'2020'}, inplace = True)
    
    df = df.query( "ts_code.str.contains('" + stock_code + "')")                
   
    return df

	
############### 数据合并 ###############

def cal_bps(df_bps, df_result):
    
    data2 = {"year":["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"],
        "bps":["","","","","","","","","","","","","","","","","","","",""]}
    
    df_year = DataFrame(data2, dtype='int64')
    
    df_year.set_index("year")
    df_cal_bps = merge( df_year, df_result, on="year", how="left")    
    
    for x in df_cal_bps['year']:
        df_cal_bps.loc[df_cal_bps.year == x, 'bps']= df_bps[str(x)].values.astype(float).round(2)
         
    df_cal_bps = df_cal_bps.round(2)
    
    return df_cal_bps
	

def cal_roe(df_roe, df_result):
    
    data2 = {"year":["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"],
        "roe":["","","","","","","","","","","","","","","","","","","",""]}
    
    df_year = DataFrame(data2, dtype='int64')
    
    df_year.set_index("year")
    df_cal_roe = merge( df_year, df_result, on="year", how="left")
    
    for x in df_cal_roe['year']:
        df_cal_roe.loc[df_cal_roe.year == x, 'roe']= df_roe[str(x)].values.astype(float).round(2)
         
    df_cal_roe = df_cal_roe.round(2)
    
    return df_cal_roe


def cal_pe(df_pe, df_result):
    
    data2 = {"year":["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"],
        "pe":["","","","","","","","","","","","","","","","","","","",""]}
    
    df_year = DataFrame(data2, dtype='int64')
    
    df_year.set_index("year")
    df_cal_pe = merge( df_year, df_result, on="year", how="left")
    
    for x in df_cal_pe['year']:
        df_cal_pe.loc[df_cal_pe.year == x, 'pe']= df_pe[str(x)].values.astype(float).round(2)
         
    df_cal_pe = df_cal_pe.round(2)
    
    return df_cal_pe
	

def cal_eps(df_eps, df_result):
    
    data2 = {"year":["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"],
        "eps":["","","","","","","","","","","","","","","","","","","",""]}
    
    df_year = DataFrame(data2, dtype='int64')
    
    df_year.set_index("year")
    df_cal_eps = merge( df_year, df_result, on="year", how="left")
    
    for x in df_cal_eps['year']:
        df_cal_eps.loc[df_cal_eps.year == x, 'eps']= df_eps[str(x)].values.astype(float).round(2)
         
    df_cal_eps = df_cal_eps.round(2)
    
    return df_cal_eps


def cal_bonus(df_bonus, df_result):
    
    data2 = {"year":["2001","2002","2003","2004","2005","2006","2007","2008","2009","2010",
        "2011","2012","2013","2014","2015","2016","2017","2018","2019","2020"],
        "bonus":["","","","","","","","","","","","","","","","","","","",""]}
    
    df_year = DataFrame(data2, dtype='int64')
    
    df_year.set_index("year")
    df_cal_bonus = merge( df_year, df_result, on="year", how="left")
    
    for x in df_cal_bonus['year']:
        df_cal_bonus.loc[df_cal_bonus.year == x, 'bonus']= df_bonus[str(x)].values.astype(float).round(2)
         
    df_cal_bonus = df_cal_bonus.round(2)
    
    return df_cal_bonus