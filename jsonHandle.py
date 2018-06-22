import json
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

# 绝对路径
def jsonOriRead(varFileName):
    print(time.strftime('Task Read ：START TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    with open(varFileName, 'r') as f:
        data = json.load(f)
        print(time.strftime('Task Read ：END TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        return data


def jsonDataRead(varFileName):
    print(time.strftime('Task Read ：START TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    with open(varFileName, 'r') as f:
        data = json.load(f)
        print(time.strftime('Task Read ：END TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        return np.array(data)

def jsonRead(varFileName):
    print(time.strftime('Task Read ：START TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    with open(varFileName, 'r') as f:
        data = json.load(f, cls='utf-8')
        print(time.strftime('Task Read ：END TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        return data

def jsonWrite(varFileName, obj):
    print(time.strftime('Task Write ：START TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    with open(varFileName, 'w') as w:
        json.dump(list(obj), w)
    print(time.strftime('Task Write ：End TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

def jsonOriWrite(varFileName, obj):
    print(time.strftime('Task Write ：START TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    with open(varFileName, 'w') as w:
        json.dump(obj, w)
    print(time.strftime('Task Write ：End TIME:' + '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
