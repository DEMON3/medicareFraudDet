import time

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
import jsonHandle

from sklearn.ensemble import IsolationForest

# needNormal 表示数据是否要做归一化处理
def DoIsolationForest(oridata,needNmormal = False):
    print('Task IsolationForest : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    if(needNmormal):
        oridata = preprocessing.minmax_scale(oridata)
    clf = IsolationForest(n_estimators=300, max_samples=256, contamination=0.15, bootstrap=True)
    clf.fit(oridata)
    output = clf.decision_function(oridata)
    print(len(output))
    print('Task IsolationForest : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return output

def getTopIndex(data, per,needNormal):
    output = DoIsolationForest(data, needNmormal=needNormal)
    res = np.argsort(np.transpose(output))
    resD = np.sort(np.transpose(output))
    lenth = len(data)
    topNum = int(np.floor(per * lenth)) + 1
    return res[:topNum]

def myIFFun(inst):
    oriData = jsonHandle.jsonDataRead('no/' + inst + '/trainingListAvg1.json')
    label = jsonHandle.jsonDataRead('no/' + inst + '/labelReal.json')

    xx=getTopIndex(oriData,1,True)
    print(xx)

# myIFFun('s062901')
