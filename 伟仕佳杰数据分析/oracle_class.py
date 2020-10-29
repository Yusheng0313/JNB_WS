import os
import cx_Oracle # 导入数据库
import pandas as pd #导入操作数据集工具
#from sqlalchemy import create_engine #导入 sqlalchemy 库,然后建立数据库连接
import time #导入时间模块
import numpy as np #导入numpy数值计算扩展


class OpOracle(object):  # 新式类
    def __init__(self, host='172.30.10.180', port='1521', sid='bpmtest', user='ecology', password='bpmtest01'):

        try:
            dsn = cx_Oracle.makedsn(host, port, sid)
            # scott是数据用户名，tiger是登录密码（默认用户名和密码）
            self.conn = cx_Oracle.connect(user, password, dsn)
        except Exception as e:
            print('数据库连接异常！%s' % e)
            quit()
        else:  # 没有异常的情况下，建立游标
            self.cur = self.conn.cursor()

    def execute(self, sql):
        try:
            self.cur.execute(sql)
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
        if sql[:6].upper() == 'SELECT':
            # return self.cur.fetchall()
            des = self.cur.description
            columns = [x[0] for x in des]
            cursor01 = self.cur.fetchall()
            data = pd.DataFrame(cursor01, columns=columns)
            return data
        else:  # 其他sql语句的话
            self.conn.commit()
            return 'ok'

    def query(self, sql):
        try:
            data = pd.read_sql_query(sql,self.conn)
        except Exception as e:
            print('sql语句有错误！%s' % e)
            return e
        else:
            return data

    def __del__(self):
        self.cur.close()
        self.conn.close()


#bpm_Op = OpOracle('172.30.10.180', '1521', 'bpmtest', 'ecology', 'bpmtest01')  # 实例化
#print(bpm_Op.query('select * from hrmresource'))
#print('ok!')
#print(bpm_Op.execute('select * from hrmdepartment'))
#print('ok!!')
#bpm_Op.__del__



# def query(table):
#
#     host = "172.30.10.180"    #数据库ip
#     port = "1521"     #端口
#     sid = "bpmtest"    #数据库名称
#     dsn = cx_Oracle.makedsn(host, port, sid)
#
#     #scott是数据用户名，tiger是登录密码（默认用户名和密码）
#     conn = cx_Oracle.connect("ecology", "bpmtest01", dsn)
#
#     #SQL语句，可以定制，实现灵活查询
#     sql = 'select * from '+ table
#
#     # 使用pandas 的read_sql函数，可以直接将数据存放在dataframe中
#     results = pd.read_sql(sql,conn)
#
#     conn.close
#     return results
#
# test_data = query('hrmresource') # 可以得到结果集
# print(test_data)

