# coding: utf-8
import pandas as pd
from pandas import DataFrame
import talib

def build_mode(df, stock_code, stock_name, startDate, endDate):
        
    if df.empty ==True:
        print(" df is empty ")
        sys.exit(2)
 
    #df = df[ df['trade_date'] > '2020-01-01']
    if len(df) <10:
        print(" len(df) <10 ")
        sys.exit(2)

    df['ma10'] = df['close'].rolling(window=10).mean()
    df.index = pd.to_datetime(df.trade_date)
    dw = pd.DataFrame()
    dw['trade_date'] = df.trade_date
    dw['trade_code'] = stock_code
    dw['name'] = stock_name
    #  baike.baidu.com/item/rsi顺势指标
    dw['rsi6'] = talib.RSI(df.close, timeperiod=6)
    dw['rsi12'] = talib.RSI(df.close, timeperiod=12)
    dw['rsi24'] = talib.RSI(df.close, timeperiod=24)
    #print("rsi6={0:.1f} , rsi12={1:.1f}, rsi24={1:.1f}".format(dw['rsi6'][-1], dw['rsi12'][-1], dw['rsi24'][-1]))
    
    return dw


def show_save_image(dw, stock_code, stock_name):
    
    import plotly.graph_objects as go
    import plotly.io as pio
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['rsi6'], mode='lines',
            name='rsi6',
            connectgaps=True,
        ))
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['rsi12'], mode='lines',
            name='rsi12',
            connectgaps=True,
        ))
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['rsi24'], mode='lines',
            name='rsi24',
            connectgaps=True,
        ))

    fig.update_layout(
        title_text = stock_code + " " +stock_name,
        #height = 300,width = 900,    
        #margin = {'t':50, 'b':50, 'l':50}
    )
    fig.show()
    
    #图片保存本地， 1.png 即保存到当前目录  D:\\指定路径     
    #pio.write_image(fig, 'D:\\1.png')
    
    return fig