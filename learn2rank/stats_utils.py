import numpy as np
import pandas as pd
from scipy.stats import entropy


def q10(x):
    return pd.Series(x).quantile(0.1)

def q20(x):
    return pd.Series(x).quantile(0.2)

def q25(x):
    return pd.Series(x).quantile(0.25)

def q30(x):
    return pd.Series(x).quantile(0.3)

def q40(x):
    return pd.Series(x).quantile(0.4)

def q60(x):
    return pd.Series(x).quantile(0.6)

def q70(x):
    return pd.Series(x).quantile(0.7)

def q75(x):
    return pd.Series(x).quantile(0.75)

def q80(x):
    return pd.Series(x).quantile(0.8)

def q90(x):
    return pd.Series(x).quantile(0.9)

from scipy.stats import entropy    
def entropy_values(x):
    x = pd.Series(x)
    return entropy(x.value_counts() / x.shape[0])
    

def kurt(x):
    return pd.Series(x).kurt()


def mode(x):
    '''
    return the mode of x, if the mode is not unique, return the first one in the sequence.
    sample：
        a = [1,1,2,2,3]# 1
        b = [1,2,2,3]  # 2
        mode(a)
        mode(b)

    '''
    return pd.Series(x).mode()[0]

from scipy.stats import moment
def fourth_moment(x):
    '''sample:
        fourth_moment([1,2,3])#2.3907061857313376
    '''
    return moment(x,4)


def fifth_moment(x):
    '''sample:
        fifth_moment([1,2,3])#2.4703447490385586
    '''
    return moment(x,5)


def range_value(x):
    '''
    range(极差) of x,
    '''
    return np.max(x)-np.min(x)


def root_mean_square(x):
    '''
    root_mean_square
    sample:
        root_mean_square([1,2,3])#2.160246899469287
    '''
    return np.sqrt(np.mean(np.array(x) ** 2))

def getIntegrationOfPSD(x_):
    '''
    
    :param x_: list data
    :return: 
    '''
    x = np.array(x_)
    avg = np.sum(x) / len(x)
    fs = 5 * len(x)# 加窗5倍
    xn = x - avg * np.ones((len(x),))
    xn = (xn - np.sum(xn) / len(xn) * np.ones((len(xn),)))
    nfft = np.ceil(np.log(fs) / np.log(2))
    pxx, f = plt.psd(xn, NFFT=nfft, Fs=fs, window=mlab.window_none, pad_to=512,
                     scale_by_freq=True)
    integration = pxx[1:len(pxx)] * (f[1:len(f)] - f[0:(len(f) - 1)])
    pow = np.sum(integration)
    return pow

all_stat_funtion = ['min', 'max', 'mean','sum', 'median',"skew",kurt, 'mad','std','var',\
                    q10, q20, q25, q30, q40, q60, q70, q75, q80, q90,\
                    entropy_values,range_value,root_mean_square, len]
all_stat_name = ['min', 'max', 'mean','sum','median',"skew",'kurt', 'mad','std','var',\
                 'q10', 'q20', 'q25', 'q30', 'q40', 'q60', 'q70', 'q75', 'q80', 'q90',\
                 'entropy_values',"range_value","root_mean_square", 'length']
