import math
import numpy as np
import matplotlib as plt

def getStatRes(data, slice):
    y = np.zeros(slice)
    for i in range(len(data)):
        cur = math.floor((data[i]-0.0001)*slice)
        y[cur] = y[cur]+1
    return y

def getTopOutlierIndexDesc(data, per):
    res = np.argsort(-np.transpose(data))
    resD = np.sort(-np.transpose(data))
    lenth = len(data)
    topNum = int(np.floor(per*lenth))+1
    if(len(res)<=1):
        res = res[0]
    return res[:topNum]

def getTopOutlierIndexAsc(data, per):
    res = np.argsort(np.transpose(data))
    resD = np.sort(np.transpose(data))
    lenth = len(data)
    topNum = int(np.floor(per*lenth))+1
    res = res[0]
    return res[:topNum]