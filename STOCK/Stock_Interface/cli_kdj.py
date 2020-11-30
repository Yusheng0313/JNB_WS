# coding: utf-8
import pandas as pd
from pandas import DataFrame
import talib

def build_mode(df, stock_code, stock_name, startDate, endDate):
    
    # 同花顺 KDJ 算法 , 
    # rolling(9, min_periods=9).min() 是每9个数为一个窗口取最小值，不足9个为NaN
    # expanding().min() 当前行的最小值
    low_list = df['low'].rolling(9, min_periods=9).min()
    low_list.fillna(value = df['low'].expanding().min(), inplace = True)
    high_list = df['high'].rolling(9, min_periods=9).max()
    high_list.fillna(value = df['high'].expanding().max(), inplace = True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    
    # ewm(com=2).mean() 指数加权移动平均线
    dw = pd.DataFrame()
    dw['trade_date'] = df['trade_date']
    dw['ts_code'] = df['ts_code']
    dw['name'] = stock_name
    dw['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    dw['D'] = dw['K'].ewm(com=2).mean()
    dw['J'] = 3 * dw['K'] - 2 * dw['D']
    dw.index = dw.trade_date

    return dw


def show_save_image(dw, stock_code, stock_name):
    
    import plotly.graph_objects as go
    import plotly.io as pio
 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['K'], mode='lines',
            name='K',
            #line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['D'], mode='lines',
            name='D',
            connectgaps=True,
        ))
    fig.add_trace(go.Scatter(x=dw['trade_date'], y=dw['J'], mode='lines',
            name='J',
            connectgaps=True,
        ))

    fig.update_layout(
        title_text = stock_code + " " +stock_name,
        #height = 300,width = 900,    
        #margin = {'t':50, 'b':50, 'l':50}
    )
    fig.show()
    
#     fig,axes = plt.subplots(2,1)
#     df[['close']].plot(ax=axes[0], grid=True, title=stock_code+" "+stock_name, figsize=(16,7))
#     # 画 KDJ 曲线图
#     dw[['K','D','J']].plot(ax=axes[1], grid=True)
#     plt.show()
    
    #图片保存本地， 1.png 即保存到当前目录  D:\\指定路径     
    #pio.write_image(fig, 'D:\\1.png')
    
    return fig