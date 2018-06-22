import math
import time

import numpy as np
from sklearn.decomposition import PCA

import jsonHandle
import myLof
import statUtil
from sklearn import preprocessing


def DoPCA(varN, pcaData, white=False):
    # do pca
    print('Task PCA : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 预处理
    pcaData = preprocessing.scale(pcaData)
    pca = PCA(n_components=varN, whiten=white, svd_solver='auto')
    pca.fit(pcaData)
    res = pca.transform(pcaData)
    print('explained variance ratio : %s'
          % str(pca.explained_variance_ratio_))
    print(pca.n_components_)
    print(np.shape(pca.components_))
    print(np.shape(pca.singular_values_))
    print('Task PCA : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return res

def DoPCA2(varN, pcaData, white=False):
    # do pca
    print('Task PCA : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # 预处理
    # pcaData = preprocessing.scale(pcaData)
    pca = PCA(n_components=varN, whiten=white, svd_solver='auto')
    pca.fit(pcaData)
    res = pca.transform(pcaData)

    print('explained variance ratio : %s'
          % str(pca.explained_variance_ratio_))
    print(pca.n_components_)
    print(np.shape(pca.components_))
    print(np.shape(pca.singular_values_))
    res = pca.inverse_transform(res)
    print('Task PCA : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return res

def dataEncode(oriData,white, stay=0.95):
    res = DoPCA(stay,oriData,white=white)
    return res

def dataEncode2(oriData,white, stay=0.85):
    res = DoPCA2(stay,oriData,white=white)
    return res

def getDoPCALOFIndex(oriData, per,white, stay=0.95 ):
    res = DoPCA(stay, oriData, white=white)
    # doLof
    result = myLof.getTopIndex(res, per, needNormal=False)
    return result

def CalOutlierScore(varN, pcaData,per):
    # do pca Cal Score
    print('Task PCA outlier Score: START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    pcaData = preprocessing.scale(pcaData)
    pca = PCA(n_components=varN, svd_solver='auto')
    pca.fit(pcaData)
    res = pca.transform(pcaData) # 1395*94
    n = pca.n_components_
    score = np.zeros(len(res))
    w = pca.components_ # 94*331
    singular = pca.singular_values_
    for i in range(len(score)):
        for j in range(math.floor(len(w)*per)):
            curw = np.mat(w[j])
            curx = np.mat(pcaData[i])
            score[i] += (curx*curw.T/singular[j])
    print('Task PCA outlier Score : END TIME:' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return score


def RebuildScore(varN, pcaData,per):
    # do pca
    print('Task PCA : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    pcaData = preprocessing.scale(pcaData)
    pca = PCA(n_components=varN, svd_solver='auto')
    pca.fit(pcaData)
    res = pca.transform(pcaData) # 1395*94
    n = pca.n_components_
    score = np.zeros(len(res))
    w = pca.components_ # 94*331
    singular = pca.singular_values_
    sumS = sum(singular)
    wt = np.transpose(w) # 331*94
    # w[1] # 1*331
    # res[1] # 1*94
    score = np.zeros((len(res), 1))
    xhat = np.zeros(np.shape(pcaData))
    for i in range(n):
        curXhat = np.array(res[:,i]).reshape((len(res), 1))*np.array([w[i]]).reshape((1,len(w[i])))
        xhat += curXhat
        weight = sum(singular[:i+1])/sumS
        if(i >= int(n-np.floor(n*per))):
            score+=np.transpose([np.sum(np.abs(xhat-pcaData), axis=1)])*weight

    print('Task PCA : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return score



# for Single

def getPCAScoreOne(data,stay = 0.95):
    curScore = RebuildScore(stay, data, 1)
    return curScore


def getTopIndexOne(data, per, method,stay = 0.95):
    if(method is 'reconstruct'):
        curScore = RebuildScore(stay, data, 1)
    elif (method is 'distance'):
        curScore = CalOutlierScore(stay, data, 1)

    return statUtil.getTopOutlierIndexDesc(curScore, per)

def getTopIndexK(data, per,k,method,stay=0.95):
    if (method is 'reconstruct'):
        curScore = RebuildScore(stay, data, k)
    elif (method is 'distance'):
        curScore = CalOutlierScore(stay, data, k)

    return statUtil.getTopOutlierIndexDesc(curScore, per)


# for Matrix

def getPCAScoreMatrix(data,stay=0.95):
    # 0.1 - 1 score
    ScoreMatrix = []
    perList = np.linspace(0, 1, 11)
    for i in range(10):
        curPer = perList[i+1]
        curScore = RebuildScore(stay, data, curPer)
        ScoreMatrix.append(curScore)
    return ScoreMatrix




def getIndexMatrix(data, per,stay=0.95):
    perList = np.linspace(0, 1, 11)
    indexMatrix = []
    for i in range(10):
        curPer = perList[i + 1]
        curScore = RebuildScore(stay, data, curPer)
        curM = statUtil.getTopOutlierIndexDesc(curScore, per)
        indexMatrix.append(curM)
    return indexMatrix


def test(inst,stay=0.95):
    data = jsonHandle.jsonDataRead('../' + inst + '/trainingList2.json')
    print(np.shape(data))
    data = preprocessing.scale(data)

    for i in range(20):
        curStay = i*0.01+0.8
        res = DoPCA(curStay, data)
        myLof.getTopIndex(data,)
    score1 = RebuildScore(stay, data, 1)
    # score1 = DoPreProcess.minMaxScale(score1)
    score2 = RebuildScore(stay, data, 0.1)
    # score2 = DoPreProcess.minMaxScale(score2)
    # y = statUtil.getStatRes(score1,100)
    # x = range(0, 100)
    set1 = statUtil.getTopOutlierIndex(score1, 0.05)
    set2 = statUtil.getTopOutlierIndex(score2, 0.05)
    count = 0
    set1 = set(set1[0])
    set2 = set(set2[0])
    print(len(set1 & set2))

# test('s062901')

