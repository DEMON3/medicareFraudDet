import jsonHandle
import numpy as np
inst = 's062901'
oriData = jsonHandle.jsonDataRead('no/' + inst + '/trainingListAvg1.json')
label = jsonHandle.jsonDataRead('no/' + inst + '/labelReal.json')
solve = np.load("supervisedData/trueIndex.npy")
solset = set(solve)
def findData():
    newData = []
    for i in range(len(oriData)):
        if(not solve.__contains__(i)):
            newData.append(list(oriData[i]))
    return newData
def findLabel():
    newData = []
    for i in range(len(label)):
        if(not solve.__contains__(i)):
            newData.append(label[i])
    return newData
