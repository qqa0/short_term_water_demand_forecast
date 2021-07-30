# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:35:09 2019

@author: chenlei
"""
from rstl import STL
from math import sqrt
from scipy.stats import t as student_t
from statsmodels.robust.scale import mad
import pandas as pd

    

def detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=None,
                 use_decomp=True, one_tail=True,
                 upper_tail=True, verbose=False):
    """
    # Detects anomalies in a time series using S-H-ESD.
    #
    # Args:
    #	 data: Time series to perform anomaly detection on.
    #	 k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
    #	 alpha: The level of statistical significance with which to accept or reject anomalies.
    #	 num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
    #	 use_decomp: Use seasonal decomposition during anomaly detection.
    #	 one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
    #	 upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
    #	 verbose: Additionally printing for debugging.
    # Returns:
    #   A dictionary containing the anomalies (anoms) and decomposition components (stl).
    """
    
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data.columns = ["timestamp", "value"]

    num_obs = len(data)
#    data.timestamp = pd.to_datetime(data.timestamp)
    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")
          
    
     # Decompose data. This returns a univarite remainder which will be used for anomaly detection. 
    decomp = STL(data.value, freq = num_obs_per_period, s_window = "periodic",
                  s_degree=0, t_window=None, t_degree=1, l_window=None, 
                  l_degree=None, s_jump=None, t_jump=None, l_jump=None,
                  robust=True,inner=None, outer=None)
     
     # Remove the seasonal component, and the median of the data to create the univariate remainder
    d = {
        'timestamp': data.timestamp,
        'value': data.value - decomp.seasonal - data.value.median()
          }
    data2 = pd.DataFrame(d)
    data = pd.DataFrame(d)
    
    p = {
        'timestamp': data.timestamp,
        'value': pd.to_numeric(pd.Series(decomp.trend + decomp.seasonal).truncate())
          }
    p['value'].index = p['timestamp'].index
    data_decomp = pd.DataFrame(p)
    
    t = {
        'timestamp': data.timestamp,
        'value': pd.to_numeric(pd.Series(decomp.trend).truncate())
         }
    t['value'].index = t['timestamp'].index
    data_trend = pd.DataFrame(t)
    
    s = {
        'timestamp': data.timestamp,
        'value': pd.to_numeric(pd.Series(decomp.seasonal).truncate())
         }
    s['value'].index = s['timestamp'].index
    data_seasonal = pd.DataFrame(s)
    
    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs * k)
    
    if max_outliers == 0:
        raise ValueError("With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %d observations in a period, which is too few. Set a higher piecewise_median_period_weeks." % num_obs)
     
    # Define values and vectors.
    n = len(data.timestamp)
    R_idx = list(range(max_outliers))

    num_anoms = 0 
     
    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(1, max_outliers + 1):
        if one_tail:
            if upper_tail:
                ares = data.value - data.value.median()
            else:
                ares = data.value.median() - data.value
        else:
            ares = (data.value - data.value.median()).abs()

        # protect against constant time series
        data_sigma = mad(data.value)
        if data_sigma == 0:
            break

        ares = ares / float(data_sigma)

        R = ares.max()

        temp_max_idx = ares[ares == R].index.tolist()[0]

        R_idx[i - 1] = temp_max_idx

        data = data[data.index != R_idx[i - 1]]

        if one_tail:
            p = 1 - alpha / float(n - i + 1)
        else:
            p = 1 - alpha / float(2 * (n - i + 1))

        t = student_t.ppf(p, (n - i - 1))
        lam = t * (n - i) / float(sqrt((n - i - 1 + t**2) * (n - i + 1)))

        if R > lam:
            num_anoms = i

    if num_anoms > 0:
        R_idx = R_idx[:num_anoms]
    else:
        R_idx = None

    return {
        'anoms': R_idx,
        'stl': data_decomp,
        'stl_trend': data_trend,
        'stl_seasonal':data_seasonal,
        'data_median':data.value.median(),
        'R':data2
        }
     
        
     
     
     
