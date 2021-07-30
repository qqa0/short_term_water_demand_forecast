# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:45:40 2019

@author: chenlei
"""

from heapq import nlargest
import numpy as np
import pandas as pd

def format_timestamp(data):
    data.timestamp = pd.to_datetime(data.timestamp)
    return data

def get_gran(tsdf, index=0):
    col = tsdf.iloc[:,index]

    largest, second_largest = nlargest(2, col)
    gran = int(round(np.timedelta64(largest - second_largest) / np.timedelta64(1, 's')))

    if gran >= 86400:
        return "day"
    elif gran >= 3600:
        return "hr"
    elif gran >= 60:
        return "min"
    elif gran >= 1:
        return "sec"
    else:
        return "ms"
    
def time_range(data,k):
    df = pd.DataFrame(columns=['timestamp','value'])
    start_date = data.timestamp.iloc[0]
    end_date = data.timestamp.iloc[-1]
    df['timestamp'] = pd.date_range(start_date, end_date, freq =str(k)+'min')
    df = df.set_index('timestamp')
    data = data.set_index('timestamp')
    df.loc[[val for val in data.index.values if val in df.index.values],'value'] = data['value'] 
    df = df.reset_index()
    
    return df
    

