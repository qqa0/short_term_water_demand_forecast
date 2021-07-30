# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:42:39 2019

@author: chenlei
"""

def data_correction(df,expected_value):
     df1 = df.copy()
     df1.loc[[value for value in expected_value.index.values if value in df.index.values],'value'] = expected_value['value']  
     return df1

def data_merge(df,correction,anoms):
     df1 = df.copy()
     correction1 = correction.copy()
     anoms1 = anoms.copy()
     
     correction1.index = correction1.timestamp
     df1.index = df1.timestamp
     df1 = df1.drop(columns='timestamp')
     correction1 = correction1.drop(columns='timestamp')
     anoms1 = anoms1.drop(columns='timestamp') 
     
     correction1.columns = ['correction']
     
     
     df1 = df1.join([correction1,anoms1])
     df1 = df1.reset_index()
     
     
     return df1





