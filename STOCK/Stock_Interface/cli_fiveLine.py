# coding: utf-8
import pandas as pd
from pandas import DataFrame

def build_mode(df, stock_code, stock_name, startDate, endDate):
       
    # 线性回归
    from sklearn import linear_model
    import numpy as np
    reg = linear_model.LinearRegression()

    #对应序号是 range(len(data))
    df['itx'] =[i  for i in range(1,len(list(df['close']))+1)]
    # x , y
    reg.fit (df['itx'].values.reshape(-1, 1),df['close'])

    #斜率为
    #print(reg.coef_)

    #截距为
    #print(reg.intercept_)
    df['coef'] = reg.coef_[0]
    df['intercept'] = reg.intercept_

    # y = c+x*b = 截距+x*斜率
    #趋势线
    df['priceTL'] = df['intercept']+df['itx']*df['coef']
    #误差
    df['y-TL'] = df['close']-df['priceTL']
    # 标准差
    df['SD'] = df['y-TL'].std()
    # 分别计算上下 1个和2个标准差
    df['TL-2SD'] = df['priceTL']-2*df['SD']
    df['TL-SD'] = df['priceTL']-df['SD']
    df['TL+2SD'] = df['priceTL']+2*df['SD']
    df['TL+SD'] = df['priceTL']+df['SD']
    df['name'] = stock_name
    
    return df


def show_save_image(df, stock_code, stock_name):
    
    import plotly.graph_objects as go
    import plotly.io as pio
  
    fig = go.Figure()
    for i in ['close','priceTL','TL-2SD', 'TL-SD', 'TL+SD', 'TL+2SD']:
        fig.add_trace(
            go.Scatter(
                x=df['trade_date'],
                y=df[i],
                name=i
                )
        )

    fig.update_layout(
        xaxis_title="年月",
        yaxis_title="单价",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        title={'text': stock_code+" "+stock_name,'xanchor': 'center','y':0.950,
            'x':0.5,
            'yanchor': 'top'},
    )

    fig.show()
    
    #图片保存本地， 1.png 即保存到当前目录  D:\\指定路径     
    pio.write_image(fig, 'D:\\1.png')
    
    return fig