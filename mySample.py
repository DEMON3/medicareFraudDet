import random
import jsonHandle
import numpy as np



def departData():
    oriData = jsonHandle.jsonDataRead('no/s062901/trainingListAvg1.json')
    label = jsonHandle.jsonDataRead('no/s062901/labelReal.json')
    sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
    sampleReal = set(sampleRes)
    falseExample = []
    trueExample = []
    count=0

    for i in range(len(oriData)):
        if(sampleReal.__contains__(label[i])):
            trueExample.append(oriData[i])
        else:
            falseExample.append(oriData[i])

    np.save('supervisedData/trueExample.npy',trueExample)
    np.save('supervisedData/falseExample.npy',falseExample)

def randomSample(k,data):
    sample = set()
    while len(sample) < k:
        cur = random.randint(0, len(data)-1)
        sample.add(cur)
    index = list(sample)
    sampleData= []
    for i in range(k):
        sampleData.append(list(data[index[i]]))
    sampleData = np.array(sampleData)
    return sampleData

departData()
