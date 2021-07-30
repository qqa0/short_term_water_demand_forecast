# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:37:30 2020

@author: chenlei
"""

from collections import namedtuple
from pandas import DataFrame, Timestamp
from detect_anoms import detect_anoms
from data_utils import format_timestamp,get_gran,time_range
from data_correction import data_correction,data_merge
# from impution.na_mean import na_mean
#from impution.na_kalman import na_kalman
import matplotlib.dates as mdates
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])


def detect_ts(df, max_anoms=0.10, direction='both',k=15, 
              alpha=0.05, threshold=None,interval=100,
              e_value=True, longterm=True,
              piecewise_median_period_weeks=2, 
              verbose=False):
    
    if list(df.columns.values) != ["timestamp", "value"]:
        df.columns = ["timestamp", "value"]

    # for i in range(len(df)):
    #     df['timestamp'][i] = df['timestamp'][i][:-3]
        
    df = format_timestamp(df)    
    df = time_range(df,k)
    df = df.fillna(0)

    gran = get_gran(df)
    mini = int(60/k * 24)
    
    gran_period = {
        'min': mini,
        'hr': 24,
        'day': 7
    }    
    
    period = gran_period.get(gran)
    
    num_obs = len(df.value)
    
    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp
        
    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iloc[-1]

        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iloc[j]
            end_date = min(start_date
                           + datetime.timedelta(days=num_days_in_period),
                           df.timestamp.iloc[-1])

            # if there is at least 14 days left, subset it,
            # otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date)
                            & (df.timestamp < end_date)]
            else:
                sub_df = df[(df.timestamp >
                     (last_date - datetime.timedelta(days=num_days_in_period)))
                    & (df.timestamp <= last_date)]
            all_data.append(sub_df)
    else:
        all_data = [df]
        
    all_anoms = DataFrame(columns=['timestamp', 'value'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'value'])
    all_data_trend = DataFrame(columns=['timestamp', 'value'])
    all_data_seasonal = DataFrame(columns=['timestamp', 'value'])
    all_R =  DataFrame(columns=['timestamp', 'value'])

    # Detect anomalies on all data (either entire data in one-pass,
    # or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]
        
        # detect_anoms actually performs the anomaly detection and
        # returns the results in a list containing the anomalies
        # as well as the decomposed components of the time series
        # for further analysis.

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms, alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)
        # store decomposed components in local variable and overwrite
        # s_h_esd_anoms to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_anoms = s_h_esd_timestamps['anoms']
        data_trend = s_h_esd_timestamps['stl_trend']
        data_seasonal = s_h_esd_timestamps['stl_seasonal']
        R = s_h_esd_timestamps['R']
#        data_median = s_h_esd_timestamps['data_median']
        
        
        # Use detected anomaly timestamps to extract the actual
        # anomalies (timestamp and value) from the data
        if s_h_esd_anoms:
            anoms = all_data[i][all_data[i].index.isin(s_h_esd_anoms)]
        else:
            anoms = DataFrame(columns=['timestamp', 'value'])
       
        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            if direction == 'both' :
            
                # Calculate daily max values
                periodic_maxes = df.groupby(
                        df.timestamp.map(Timestamp.date)).aggregate(np.max).value
                periodic_mins = df.groupby(
                        df.timestamp.map(Timestamp.date)).aggregate(np.min).value

                # Calculate the threshold set by the user
                if threshold == 'med':
                    thresh1 = periodic_maxes.median()
                    thresh2 = periodic_mins.median()
                elif threshold == 'p95':
                    thresh1 = periodic_maxes.quantile(.95)
                    thresh2 = periodic_mins.quantile(.05)
                elif threshold == 'p99':
                    thresh1 = periodic_maxes.quantile(.99)
                    thresh2 = periodic_mins.quantile(.01)
                    # Remove any anoms below the threshold
                    anoms1 = anoms[anoms.value >= thresh1]
                    anoms2 = anoms[anoms.value <= thresh2]
                    anoms = pd.concat([anoms1,anoms2])
                    
            if direction == 'pos' :
                # Calculate daily max values
                periodic_maxes = df.groupby(
                        df.timestamp.map(Timestamp.date)).aggregate(np.max).value
                # Calculate the threshold set by the user
                if threshold == 'med':
                    thresh = periodic_maxes.median()                    
                elif threshold == 'p95':
                    thresh = periodic_maxes.quantile(.95)                   
                elif threshold == 'p99':
                    thresh = periodic_maxes.quantile(.99)
                   
                    # Remove any anoms below the threshold
                    anoms1 = anoms[anoms.value >= thresh]  
            if direction == 'neg' :
                # Calculate daily max values
                periodic_mins = df.groupby(
                        df.timestamp.map(Timestamp.date)).aggregate(np.min).value

                # Calculate the threshold set by the user
                if threshold == 'med':
                    thresh = periodic_mins.median()
                elif threshold == 'p95':
                    thresh = periodic_mins.quantile(.05)
                elif threshold == 'p99':
                    thresh = periodic_mins.quantile(.01)
                    # Remove any anoms below the threshold
                    anoms2 = anoms[anoms.value <= thresh]                                              

#        all_anoms = all_anoms.append(anoms1)
        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(data_decomp)
        all_data_trend = all_data_trend.append(data_trend)
        all_data_seasonal = all_data_seasonal.append(data_seasonal)
        all_R = all_R.append(R)

    # Cleanup potential duplicates
    try:
        all_anoms.drop_duplicates(subset=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(subset=['timestamp'], inplace=True)
        all_data_trend.drop_duplicates(subset=['timestamp'], inplace=True)
        all_data_seasonal.drop_duplicates(subset=['timestamp'], inplace=True)
        all_R.drop_duplicates(subset=['timestamp'], inplace=True)
    except TypeError:
        all_anoms.drop_duplicates(cols=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(cols=['timestamp'], inplace=True)
        all_data_trend.drop_duplicates(cols=['timestamp'], inplace=True)
        all_data_seasonal.drop_duplicates(cols=['timestamp'], inplace=True)
        all_R.drop_duplicates(cols=['timestamp'], inplace=True)

    all_anoms.index = all_anoms.timestamp
   
    if e_value:
        expected_value = seasonal_plus_trend[seasonal_plus_trend.timestamp.isin(all_anoms.timestamp)]
        correction = data_correction(df,expected_value)
        expected_value = expected_value.set_index('timestamp')
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value,
            'expected_value': expected_value.value
        }
    else:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value
        }
    anoms = DataFrame(d, index=d['timestamp'].index)    
    
    merge = data_merge(df,correction,anoms)
    
    return df,anoms,all_data_seasonal,all_data_trend,all_R,correction,merge



    
def data_plot(df,anoms,freq='1min',interval=100,xlabel='date',
              ylabel='data'):
    
    anoms_label = plt.figure() #异常值标注
    start_date = df.timestamp.iloc[0]
    end_date = df.timestamp.iloc[-1]
    dates = pd.date_range(start_date, end_date, freq = freq)
    xs = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    label = ['raw_data','anoms']
    plt.plot(xs, df.value, linewidth='0.5', color='b')
    plt.scatter(anoms.timestamp,  anoms.anoms,c='r',marker='*')
    plt.legend(label,loc='best')
    plt.gcf().autofmt_xdate()
    plt.xlabel(xlabel, family='Times New Roman', fontsize=12)  # X轴
    plt.ylabel(ylabel, family='Times New Roman', fontsize=12)  # Y轴
    plt.title('Outlier location', family='Times New Roman', fontsize=12)    
    
    seasonal_plot = plt.figure() #季节性图
    start_date = df.timestamp.iloc[0]
    end_date = df.timestamp.iloc[-1]
    dates = pd.date_range(start_date, end_date, freq = freq)
    xs = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    label = ['seasonal']
    plt.plot(xs, all_data_seasonal.value, linewidth='0.5', color='b')
    plt.legend(label, loc='best')
    plt.gcf().autofmt_xdate()
    plt.xlabel(xlabel, family='Times New Roman', fontsize=12)  # X轴
    plt.ylabel(ylabel, family='Times New Roman', fontsize=12)  # Y轴
    plt.title('Seasonal', family='Times New Roman', fontsize=12)    
    
    trend_plot = plt.figure() #趋势图及中位数

    start_date = df.timestamp.iloc[0]
    end_date = df.timestamp.iloc[-1]
    dates = pd.date_range(start_date, end_date, freq = freq)
    xs = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    label = ['trend']
    plt.plot(xs, all_data_trend.value, linewidth='0.5', color='b')
    plt.legend(label, loc='best')
    plt.gcf().autofmt_xdate()
    plt.xlabel(xlabel, family='Times New Roman', fontsize=12)  # X轴
    plt.ylabel(ylabel, family='Times New Roman', fontsize=12)  # Y轴
    plt.title('Trend', family='Times New Roman', fontsize=12)    
    
    R_plot = plt.figure()
    start_date = df.timestamp.iloc[0]
    end_date = df.timestamp.iloc[-1]
    dates = pd.date_range(start_date, end_date, freq = freq)
    xs = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    label = ['R']
    plt.plot(xs, all_R.value, linewidth='0.5', color='b')
    plt.legend(label, loc='best')
    plt.gcf().autofmt_xdate()
    plt.xlabel(xlabel, family='Times New Roman', fontsize=12)  # X轴
    plt.ylabel(ylabel, family='Times New Roman', fontsize=12)  # Y轴
    plt.title('R', family='Times New Roman', fontsize=12)    
    
    correction_plot = plt.figure() #修正值
    start_date = df.timestamp.iloc[0]
    end_date = df.timestamp.iloc[-1]
    dates = pd.date_range(start_date, end_date, freq = freq)
    xs = [datetime.datetime.strptime(str(d), '%Y-%m-%d %H:%M:%S') for d in dates]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    label = ['correction']
    plt.plot(xs, correction.value, linewidth='0.5', color='b')
    plt.legend(label, loc='best')
    plt.gcf().autofmt_xdate()
    plt.xlabel(xlabel, family='Times New Roman', fontsize=12)  # X轴
    plt.ylabel(ylabel, family='Times New Roman', fontsize=12)  # Y轴
    plt.title('Data Correction', family='Times New Roman', fontsize=12)
                  

if __name__ == '__main__':

    df = pd.read_excel('xx.xls')

    df,anoms,all_data_seasonal,all_data_trend,all_R,correction,merge = detect_ts(df,
                  max_anoms=0.10, direction='both',k=1,
                  alpha=0.05, threshold=None,interval=100,
                  e_value=True, longterm=False,
                  piecewise_median_period_weeks=2,
                  verbose=False)

    data_plot(df,anoms,freq='1min',interval=100,xlabel='date',
                  ylabel='data')

    merge.to_csv('result.csv')



