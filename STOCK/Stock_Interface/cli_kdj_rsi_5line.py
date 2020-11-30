# coding: utf-8
import pandas as pd
from pandas import DataFrame
import talib
from Stock_Interface import cli_fiveLine
from Stock_Interface import cli_rsi
from Stock_Interface import cli_kdj
pd.options.plotting.backend = "plotly"
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def show_fig(df, stock_code, stock_name, startDate, endDate):
    
    df_fiveline = cli_fiveLine.build_mode(df, stock_code, stock_name, startDate, endDate)    
    df_rsi = cli_rsi.build_mode(df, stock_code, stock_name, startDate, endDate)    
    df_kdj = cli_kdj.build_mode(df, stock_code, stock_name, startDate, endDate)
    

    fig = make_subplots(rows=4, cols=1)
    #fig.append_trace(go.Scatter( x=df['trade_date'], y=df['close'],) , row=1, col=1)
    
    for i in ['priceTL','TL-2SD', 'TL-SD', 'TL+SD', 'TL+2SD']:
        fig.add_trace(go.Scatter( x=df_fiveline['trade_date'], y=df_fiveline[i], name=i ) , row=1, col=1)
    
#     fig = go.Figure(data=[go.Candlestick(x=df_fiveline['trade_date'],
#                 open=df_fiveline['Open'], high=df_fiveline['High'],
#                 low=df_fiveline['Low'], close=df_fiveline['Close']) 
#                       ])
    fig.append_trace(go.Candlestick( x=df_fiveline['trade_date'],
                open=df_fiveline['open'], high=df_fiveline['high'],
                low=df_fiveline['low'], close=df_fiveline['close'], name='price') , row=1, col=1)
    
    fig.append_trace(go.Scatter( x=df_kdj['trade_date'], y=df_kdj['K'], name='K') , row=3, col=1)
    fig.append_trace(go.Scatter( x=df_kdj['trade_date'], y=df_kdj['D'], name='D') , row=3, col=1)
    fig.append_trace(go.Scatter( x=df_kdj['trade_date'], y=df_kdj['J'], name='J') , row=3, col=1)
    
    fig.append_trace(go.Scatter( x=df_rsi['trade_date'], y=df_rsi['rsi6'], name='rsi6') , row=4, col=1)

    fig.update_layout(height=650, width=1000, 
                      xaxis_rangeslider_visible=False,
                      title_text=stock_code+' '+stock_name)
    fig.show()
    
    return fig

