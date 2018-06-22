import random
import jsonHandle
import numpy as np


sample=set()
while len(sample)<400:
    cur = random.randint(0, 1394)
    sample.add(cur)

jsonHandle.jsonWrite('random400.json', sample)

oriData = jsonHandle.jsonDataRead('no/s062901/trainingListAvg1.json')
label = jsonHandle.jsonDataRead('no/s062901/labelReal.json')
sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
sampleReal = set(sampleRes)
testList400=[]
trainingList=[]
labelTest=[]
labelTrain=[]
classTrain=[]
classTest=[]
indexList = list(sample)
count=0

for i in range(len(oriData)):
    if(sample.__contains__(i)):
        testList400.append(oriData[i])
        labelTest.append(label[i])
        if(sampleReal.__contains__(label[i])):
            count+=1
            classTest.append(1)
        else:
            classTest.append(0)
    else:
        trainingList.append(oriData[i])
        labelTrain.append(label[i])
        if (sampleReal.__contains__(label[i])):
            classTrain.append(1)
        else:
            classTrain.append(0)


np.save('supervisedData/sampleIndex.npy',indexList)
np.save('supervisedData/testList400.npy',testList400)
np.save('supervisedData/labelTest.npy',labelTest)
np.save('supervisedData/trainingList.npy',trainingList)
np.save('supervisedData/labelTrain.npy',labelTrain)
np.save('supervisedData/classTrain.npy',classTrain)
np.save('supervisedData/classTest.npy',classTest)
print(count)


