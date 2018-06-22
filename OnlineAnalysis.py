import numpy as np
import myLR
import mySample
import math
import random
import matplotlib.pyplot as plt
def mainPro3():
    train = np.load('supervisedData/trainingList.npy')
    test = np.load('supervisedData/testList400.npy')
    labelTest = np.load('supervisedData/labelTest.npy')
    classTest = np.load('supervisedData/classTest.npy')
    classTrain = np.load('supervisedData/classTrain.npy')
    t = 0.30
    trueE = np.load('supervisedData/trueExample.npy')
    falseE = np.load('supervisedData/falseExample.npy')
    falseE = mySample.randomSample(math.ceil((len(trueE) / t) * (1 - t)), falseE)
    # falseE = np.ndarray(falseE)
    trueELabel = np.ones(trueE.shape[0])
    falseELabel = np.zeros(falseE.shape[0])
    test = np.load('supervisedData/testList400.npy')
    classTest = np.load('supervisedData/classTest.npy')
    print('-----------' + str(t) + '------------')
    res = myLR.DoLR(np.append(trueE, falseE, axis=0), np.append(trueELabel, falseELabel, axis=0), test)
    res = resultCompare(res, classTest)



def mainPro2():
    trueE = np.load('supervisedData/trueExample.npy')
    falseE = np.load('supervisedData/falseExample.npy')
    t=0.6
    x = np.linspace(0.15,0.60,46)
    result = []
    for i in range(len(x)):
        t = x[i]
        falseE = mySample.randomSample(math.ceil((len(trueE)/t)*(1-t)),falseE)
       # falseE = np.ndarray(falseE)
        trueELabel = np.ones(trueE.shape[0])
        falseELabel = np.zeros(falseE.shape[0])
        test = np.load('supervisedData/testList400.npy')
        classTest = np.load('supervisedData/classTest.npy')
        print('-----------'+str(t)+'------------')
        res = myLR.DoLR(np.append(trueE,falseE,axis=0),np.append(trueELabel,falseELabel,axis=0),test)
        res = resultCompare(res,classTest)
        result.append(res.get('f'))
    plt.figure()
    plt.plot(x,result)
    plt.xlabel('positive proportion')
    plt.ylabel('F score')
    plt.show()


def mainPro():
    train = np.load('supervisedData/trainingList.npy')
    test = np.load('supervisedData/testList400.npy')
    labelTest = np.load('supervisedData/labelTest.npy')
    classTest = np.load('supervisedData/classTest.npy')
    classTrain = np.load('supervisedData/classTrain.npy')
    res = myLR.DoLR(train, classTrain, test)
    res2 = myLR.DoLR(train, classTrain, train)
    resultCompare(res, classTest)
    resultCompare(res2, classTrain)

def resultCompare(res,classTest):
    count =0
    sum =0
    real=0
    labelI = np.load("supervisedData/sampleIndex.npy")
    trueIndex = []
    for i in range(len(classTest)):
        if(classTest[i] == 1):
            real+=1
        if(res[i]==1) :
            sum+=1
            trueIndex.append(labelI[i])
        if(classTest[i]==1 and classTest[i] == res[i]):
            count+=1
    print('-----------------------')
    np.save("supervisedData/trueIndex.npy",trueIndex)
    print(count)
    print(sum)
    print(real)
    p=count/sum
    r=count / real
    return {'p':p,
            'r': r,
            'f': (2*p*r)/(p+r)
            }

mainPro3()
