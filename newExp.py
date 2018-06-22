import random
import numpy as np
import jsonHandle
import ExProcess
import time

import numpy as np
import jsonHandle
import math
import myIF
import myLof
import myPca
import myKpca
from sklearn import preprocessing
import returnResutl
import scipy.io
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(331, 150),
            nn.Tanh()

            # nn.Linear(64, 16),
            # nn.Tanh(),
            # nn.Linear(16, 3),   # 压缩成3个特征, 进行 3D 图像可视化
        )
        # 解压
        self.decoder = nn.Sequential(
            # nn.Linear(3, 16),
            # nn.Tanh(),
            # nn.Linear(16, 64),
            # nn.Tanh(),
            nn.Linear(150, 331),
            nn.Tanh(),       # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def dataEncode(data, EPOCH = 1000, BATCH_SIZE =5, LR = 0.00001):
    oriData = data
    scData = preprocessing.minmax_scale(oriData)
    scData = torch.from_numpy(scData).type(torch.FloatTensor)
    oriSc = Variable(scData)
    train_loader = Data.DataLoader(dataset=scData, batch_size=BATCH_SIZE, shuffle=True)
    ae = AutoEncoder()
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):
        for step, x in enumerate(train_loader):
            tx = Variable(x)
            ty = Variable(x)
            encoded, decoded = ae(tx)
            loss = loss_func(decoded, ty)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 1 == 0:
                print('Epoch: ', epoch, '| train loss: %.6f' % loss.data[0])
                # print(len(loss))
    encoded_data, _ = ae(oriSc)
    return encoded_data


def dataEncodeFor(approach, oriData):
    if(approach is 'AE'):
        encodedAEData = dataEncode(oriData)
        npen = encodedAEData.data.numpy()
        np.save('aeEncode.npy', npen)
        return npen
    if (approach is 'PCA'):
        return myPca.dataEncode(oriData, white=False)
    if (approach is 'PCA2'):
        return myPca.dataEncode2(oriData, white=False,stay=0.85)
    if (approach is 'KPCA'):
        kpcaEncode =  myKpca.dataEncode('rbf', oriData, 90)
        return kpcaEncode
    if (approach is 'KPCA2'):
        kpcaEncode =  myKpca.dataEncode('rbf', oriData, 90)
        return kpcaEncode
    if (approach is 'to01'):
        return preprocessing.minmax_scale(oriData)
    if (approach is 'Normal'):
        return preprocessing.scale(oriData)

def calScore(approach, encodeData, needNormal = True):
    if(approach is 'LOF'):
        return myLof.DoLOF(encodeData, needNormal)
    if (approach is 'IF'):
        return myIF.DoIsolationForest(encodeData, False)

def getTopIndex(output):
    res = np.argsort(np.transpose(output))
    resD = np.sort(np.transpose(output))
    return res

def labelReSort(output,label, per):
    res = getTopIndex(output)
    lenth = len(output)
    topNum = int(np.floor(per * lenth)) + 1
    res = res[:topNum]
    labelRet = []
    for i in range(topNum-1):
        labelRet.append(label[res[i]])
    return labelRet

def DoForOneNormal(oriData, approachEncode, approachCal):
    # 1 DataEncode
    encodeData = dataEncodeFor(approachEncode,oriData)
    # 2 OutLier Detection
    score = calScore(approachCal,encodeData)
    # 3 return labelresort
    rsLabel = getTopIndex(score)
    return rsLabel

def getResult(k, realResult , slice, falseNum,per):
    result100 = []
    recall = []
    precision = []
    f11 = []
    tpr = []
    fpr = []
    sumCount = 0
    falseCount= 0
    tmpCount = 0
    tmpCountF = 0
    lenth = len(realResult)
    tn = lenth-falseNum
    cellLen = math.ceil(lenth / slice)
    for i in range(len(realResult)):
        if (realResult[i]<k):
            tmpCount += 1
        else:
            tmpCountF += 1
        if ((i + 1) % (cellLen) == 0):
            result100.append(tmpCount)
            sumCount += tmpCount
            falseCount += tmpCountF
            precision.append(sumCount / (i + 1))
            fpr.append(falseCount / tn)
            recall.append(sumCount / falseNum)
            cf = sumCount/((i+1)+falseNum);
            f11.append(cf)
            falseCount=0
            tmpCount = 0
    print(len(result100))
    print(result100)
    print(recall)
    print("recall 3 5 8 10 15 20 ")
    print(recall[2], recall[4], recall[7], recall[9], recall[14], recall[19])
    print(precision)
    print("precision 3 5 8 10 15 20 ")
    print(precision[2], precision[4], precision[7], precision[9], precision[14], precision[19])
    print(f11)
    print("f1 3 5 8 10 15 20 ")
    print(f11[2], f11[4], f11[7], f11[9], f11[14], f11[19])
    print(np.sum(result100[:math.ceil(slice*per)]))
    return {'precision': list(precision),
            'recall': list(recall),
            'fpr': list(fpr),
            'f1': list(f11),
            'result100': list(result100),
            'slice': len(precision),
            'realCount': float(np.sum(result100[:math.ceil(slice*per)]))
            }
def getResult2(classLabel, realResult , slice, falseNum,per):
    result100 = []
    recall = []
    precision = []
    f11 = []
    tpr = []
    fpr = []
    sumCount = 0
    falseCount= 0
    tmpCount = 0
    tmpCountF = 0
    lenth = len(realResult)
    tn = lenth-falseNum
    cellLen = math.ceil(lenth / slice)
    for i in range(len(realResult)):
        if (classLabel[realResult[i]]==1):
            tmpCount += 1
        else:
            tmpCountF += 1
        if ((i + 1) % (cellLen) == 0):
            result100.append(tmpCount)
            sumCount += tmpCount
            falseCount += tmpCountF
            precision.append(sumCount / (i + 1))
            fpr.append(falseCount / tn)
            recall.append(sumCount / falseNum)
            cf = sumCount/((i+1)+falseNum);
            f11.append(cf)
            falseCount=0
            tmpCount = 0
    print(len(result100))
    print(result100)
    print(recall)
    print("recall 3 5 8 10 15 20 ")
    print(recall[2], recall[4], recall[7], recall[9], recall[14], recall[19])
    print(precision)
    print("precision 3 5 8 10 15 20 ")
    print(precision[2], precision[4], precision[7], precision[9], precision[14], precision[19])
    print(f11)
    print("f1 3 5 8 10 15 20 ")
    print(f11[2], f11[4], f11[7], f11[9], f11[14], f11[19])
    print(np.sum(result100[:math.ceil(slice*per)]))
    return {'precision': list(precision),
            'recall': list(recall),
            'fpr': list(fpr),
            'f1': list(f11),
            'result100': list(result100),
            'slice': len(precision),
            'realCount': float(np.sum(result100[:math.ceil(slice*per)]))
            }

def randomSampleD(k,data):
    sample = set()
    lenth = len(data)
    while len(sample) < (lenth-k):
        cur = random.randint(0, lenth-1)
        sample.add(cur)
    sampleData= []
    for i in range(lenth):
        if(not sample.__contains__(i)):
            sampleData.append(list(data[i]))
    sampleData = np.array(sampleData)
    return sampleData


def randomSample(k,data):
    sample = set()
    lenth = len(data)
    while len(sample) < k:
        cur = random.randint(0, lenth-1)
        sample.add(cur)
    index = list(sample)
    sampleData= []
    for i in range(k):
        sampleData.append(list(data[index[i]]))
    sampleData = np.array(sampleData)
    return sampleData

allData = np.load('supervisedData/testList400.npy')
# allData = np.append(trainTrue,trainFalse,axis=0)

AeLOFLabel = DoForOneNormal(allData, 'AE', 'LOF')
PCALOFLabel = DoForOneNormal(allData,  'PCA', 'LOF')
KPCALOFLabel = DoForOneNormal(allData,  'KPCA', 'LOF')
LOFLabel = DoForOneNormal(allData,  'to01', 'LOF')

AeIFLabel = DoForOneNormal(allData,   'AE', 'IF')
PCAIFLabel = DoForOneNormal(allData,   'PCA', 'IF')
KPCAIFLabel = DoForOneNormal(allData, 'KPCA2', 'IF')
IFLabel = DoForOneNormal(allData,   'to01', 'IF')
testLabel = np.load('supervisedData/classTest.npy')

base = getResult2(testLabel, LOFLabel, 100, 54, 0.1)
res1 = getResult2(testLabel, AeLOFLabel, 100, 54, 0.1)
res2 = getResult2(testLabel, PCALOFLabel, 100, 54, 0.1)
res3 = getResult2(testLabel, KPCALOFLabel, 100, 54, 0.1)
result = {
    'AE': res1,
    'PCA': res2,
    'KPCA': res3,
}
ExProcess.plotShowInOne(base, result)

base1 = getResult2(testLabel, LOFLabel, 100, 54, 0.1)
res11 = getResult2(testLabel, AeLOFLabel, 100, 54, 0.1)
res21 = getResult2(testLabel, PCAIFLabel, 100, 54, 0.1)
res31 = getResult2(testLabel, KPCAIFLabel, 100, 54, 0.1)
result1 = {
    'AE': res11,
    'PCA': res21,
    'KPCA': res31,
}
ExProcess.plotShowInOne2(base1, result1)


