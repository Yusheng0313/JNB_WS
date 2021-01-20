import os
#import cx_Oracle # 导入数据库
import pandas as pd #导入操作数据集工具
import time #导入时间模块
import numpy as np #导入numpy数值计算扩展
from sqlalchemy import create_engine


class oracle_op(object):  # 新式类
    
    def __init__(self, host='172.30.10.180', port='1521', sid='bpmtest', user='ecology', password='bpmtest01'):
        try:
            conn_string = 'oracle+cx_oracle://'+user+':'+password+'@'+host+':'+port+'/'+sid
            self.conn = create_engine(conn_string, echo=False,encoding='utf-8')
        except Exception as e:
            print('数据库连接异常！%s' % e)
            quit()


    def query(self, sql):
        try:
            data = pd.read_sql(sql,self.conn)
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
        else:
            return data

    
    def append(self, tablename, data, dtype):
        try:
            # if_exists = fail,replace,append
            data.to_sql(tablename, self.conn, index=False, if_exists='append', dtype=dtype)            
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
        else:
            return data
     
    def deleteAll(self, tablename):      
        try:
            self.conn.execute('truncate table {}'.format(tablename))
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
            
        
    def execute(self, sql):
        try:
            self.conn.execute(sql)
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
    
    
    def insert(self, tablename, data):   
        
        try:
            conn = self.conn.raw_connection()
            #db = cx_Oracle.connect(orcl_engien[9:])
            cursor = conn.cursor()
            #cursor.copy_from(output, ttb, null='')
            col = ', '.join(data.columns.tolist())
            s = ', '.join([':'+str(i) for i in range(1, data.shape[1]+1)])
            sql = 'insert into {}({}) values({})'.format(tablename, col, s)
            #TypeError: expecting float, 清理该字段 NaN值 df['numberfield1'] = df['numberfield1'].fillna(0)

            cursor.executemany(sql, data.values.tolist())
            conn.commit()
            cursor.close()
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e


# bpm_Op = NewOpOracle('172.30.10.180', '1521', 'bpmtest', 'ecology', 'bpmtest01')  # 实例化
# print(bpm_Op.query('select * from hrmresource'))
# print('123!!!')

# connect = create_engine('mysql+pymysql://username:password@host:port/dbname')
#
#
# sql = 'select * from test'
# data = pd.read_sql(sql,connect)
#
# data.to_sql('tablename', connect, index=False, if_exists='append')