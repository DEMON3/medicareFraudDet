import time

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor


# needNormal 表示数据是否要做归一化处理
def DoLOF(lofData,needNmormal = True):
    print('Task LOF : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    if(needNmormal):
        lofData = preprocessing.minmax_scale(lofData)
    clf = LocalOutlierFactor(n_neighbors=20, n_jobs=3)
    y_pred = clf.fit_predict(lofData)
    output = clf.negative_outlier_factor_
    print(len(y_pred))
    print('Task LOF : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return output

def getTopIndex(data, per,needNormal):
    output = DoLOF(data, needNmormal=needNormal)
    res = np.argsort(np.transpose(output))
    np.save('ifl.npy',res)
    resD = np.sort(np.transpose(output))
    lenth = len(data)
    topNum = int(np.floor(per * lenth)) + 1
    return res[:topNum]

