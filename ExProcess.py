import time

import numpy as np
import jsonHandle
import myAE
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
import newData

class AutoEncoder(nn.Module):
    def __init__(self,stay):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(331, stay),
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
            nn.Linear(stay, 331),
            nn.Tanh(),       # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def dataEncode(data, stay, EPOCH = 500, BATCH_SIZE = 32, LR = 0.0001 ):
    oriData = data
    scData = preprocessing.minmax_scale(oriData)
    scData = torch.from_numpy(scData).type(torch.FloatTensor)
    oriSc = Variable(scData)
    train_loader = Data.DataLoader(dataset=scData, batch_size=BATCH_SIZE, shuffle=True)
    ae = AutoEncoder(stay)
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

def dataEncodeFor(approach, oriData,stay=0.95):
    if(approach is 'AE'):
        encodedAEData = dataEncode(oriData, stay if stay%1==0 else stay*331)
        npen = encodedAEData.data.numpy()
        # np.save('aeEncode.npy', npen)
        return npen
            # np.load('aeEncode.npy')
        # return npen
    if (approach is 'PCA'):
        return myPca.dataEncode(oriData, white=False,stay=stay)
    if (approach is 'PCA2'):
        return myPca.dataEncode2(oriData, white=False,stay=0.85)
    if (approach is 'KPCA'):
        kpcaEncode =  myKpca.dataEncode('rbf', oriData, stay if stay%1==0 else stay*331)
        return kpcaEncode
    if (approach is 'KPCA2'):
        kpcaEncode =  myKpca.dataEncode('rbf', oriData, stay if stay%1==0 else stay*331)
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

def DoForOneApproach(oriData,label, approachEncode, approachCal,stay=0.95):
    # 1 DataEncode
    encodeData = dataEncodeFor(approachEncode,oriData,stay=stay)
    # 2 OutLier Detection
    score = calScore(approachCal,encodeData)
    # 3 return labelresort
    rsLabel = labelReSort(score, label, 1)
    return rsLabel

def DoForOneNormal(oriData,k, approachEncode, approachCal):
    # 1 DataEncode
    encodeData = dataEncodeFor(approachEncode,oriData)
    # 2 OutLier Detection
    score = calScore(approachCal,encodeData)
    # 3 return labelresort
    rsLabel = labelReSort(score, k, 1)
    return rsLabel




def labelJudge(sampleLabel, alLabel, slice, falseNum, per):
    return returnResutl.getResult(sampleLabel, alLabel, slice, falseNum, per)

def plotShow(baseLine, result):
    plt.figure()
    l1, = plt.plot(baseLine.get('recall'), baseLine.get('precision'), 'r-')
    l2, = plt.plot(result.get('recall'), result.get('precision'), 'y-.')
    plt.legend(handles=[l1, l2, ], labels=['LOF', 'PCA-LOF' ], loc='best')
    plt.show()

def plotShowInOne(baseLine, result):
    plt.figure()
    jsonHandle.jsonOriWrite('result/lof.json', baseLine)
    jsonHandle.jsonOriWrite('result/aelof.json', result.get('AE'))
    jsonHandle.jsonOriWrite('result/pcalof.json', result.get('PCA'))
    jsonHandle.jsonOriWrite('result/kpcalof.json', result.get('KPCA'))
    l1, = plt.plot(baseLine.get('recall'), baseLine.get('precision'), 'r-')
    l2, = plt.plot(result.get('AE').get('recall'), result.get('AE').get('precision'), 'y-.')
    l3, = plt.plot(result.get('PCA').get('recall'), result.get('PCA').get('precision'), 'b--')
    l4, = plt.plot(result.get('KPCA').get('recall'), result.get('KPCA').get('precision'), 'g*-')
    plt.legend(handles=[l1, l2, l3, l4 ], labels=['LOF' , 'AE-LOF', 'PCA-LOF' , 'KPCA-LOF'], loc='best')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

def plotShowInOne2(baseLine, result):
    plt.figure()
    jsonHandle.jsonOriWrite('result/if.json', baseLine)
    jsonHandle.jsonOriWrite('result/aeif.json', result.get('AE'))
    jsonHandle.jsonOriWrite('result/pcaif.json', result.get('PCA'))
    jsonHandle.jsonOriWrite('result/kpcaif.json', result.get('KPCA'))
    l1, = plt.plot(baseLine.get('recall'), baseLine.get('precision'), 'r-')
    l2, = plt.plot(result.get('AE').get('recall'), result.get('AE').get('precision'), 'y-.')
    l3, = plt.plot(result.get('PCA').get('recall'), result.get('PCA').get('precision'), 'b--')
    l4, = plt.plot(result.get('KPCA').get('recall'), result.get('KPCA').get('precision'), 'g*-')
    plt.legend(handles=[l1, l2, l3, l4 ], labels=['IF' , 'AE-IF', 'PCA-IF' , 'KPCA-IF'], loc='best')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

def OneTimeMain(inst):
    print(time.strftime('Task Main ：START TIME:'+'%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
    task = ['LOFRes', 'pcaLOFRes', 'pcaPolyRes', 'pcaRes', 'pcaRbfRes']
    # 1 加载数据
    oriData = newData.findData()
        #jsonHandle.jsonDataRead('no/' + inst + '/trainingListAvg1.json')
    label = newData.findLabel()
    # jsonHandle.jsonDataRead('no/' + inst + '/labelReal.json')
    # Label ReSort Res
    AeLOFLabel = DoForOneApproach(oriData,label,'AE','LOF')
    PCALOFLabel = DoForOneApproach(oriData,label,'PCA','LOF')
    KPCALOFLabel = DoForOneApproach(oriData,label,'KPCA','LOF')
    LOFLabel = DoForOneApproach(oriData,label,'to01','LOF')

    AeIFLabel = DoForOneApproach(oriData, label, 'AE', 'IF')
    PCAIFLabel = DoForOneApproach(oriData, label, 'PCA', 'IF')
    KPCAIFLabel = DoForOneApproach(oriData, label, 'KPCA2', 'IF')
    IFLabel = DoForOneApproach(oriData, label, 'to01', 'IF')
    #testtest
    # print("test----------------------")
    # ifLabel = np.load('ifl.npy')
    # labelRet = []
    # for i in range(len(ifLabel) - 1):
    #     labelRet.append(label[ifLabel[i]])
    # ifBase = labelJudge(sampleRes, labelRet, 100, 205, 0.1)
    # print("test----------------------")
    base = labelJudge(sampleRes, LOFLabel, 100, 177, 0.1)
    res1 = labelJudge(sampleRes, AeLOFLabel, 100, 177, 0.1)
    res2 = labelJudge(sampleRes, PCALOFLabel, 100, 177, 0.1)
    res3 = labelJudge(sampleRes, KPCALOFLabel, 100, 177, 0.1)
    result = {
        'AE': res1,
        'PCA': res2,
        'KPCA': res3,
    }
    base1 = labelJudge(sampleRes, IFLabel, 100, 177, 0.1)
    res11 = labelJudge(sampleRes, AeIFLabel, 100, 177, 0.1)
    res21 = labelJudge(sampleRes, PCAIFLabel, 100, 177, 0.1)
    res31 = labelJudge(sampleRes, KPCAIFLabel, 100, 177, 0.1)
    result1 = {
        'AE': res11,
        'PCA': res21,
        'KPCA': res31,
    }
    plotShowInOne(base, result)
    plotShowInOne2(base1, result1)
    # plotShow(result,base,'precision')
    # plotShow(result,base,'recall')
    # plotShow(labelJudge(sampleRes,LOFLabel,100,205,0.1),labelJudge(sampleRes,AeLOFLabel,100,205,0.1))
    # plotShow(labelJudge(sampleRes,LOFLabel,100,205,0.1),labelJudge(sampleRes,PCALOFLabel,100,205,0.1))
    # plotShow(labelJudge(sampleRes,LOFLabel,100,205,0.1),labelJudge(sampleRes,KPCALOFLabel,100,205,0.1))



def mainForPublic():
    print(time.strftime('Task Main ：START TIME:'+'%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    data = scipy.io.loadmat('arrhythmia.mat')
    X = data.get('X')
    Y = data.get('y')
    task = ['LOFRes', 'pcaLOFRes', 'pcaPolyRes', 'pcaRes', 'pcaRbfRes']
    # 1 加载数据
    sampleRes = []
    for i in range(len(Y)):
        if(Y[i]>0):
            sampleRes.append(i)
    oriData = X
    label = range(len(X))
    # Label ReSort Res
    # AeLOFLabel = DoForOneApproach(oriData,label,'AE','LOF')
    PCALOFLabel = DoForOneApproach(oriData,label,'PCA','LOF')
    PCA2LOFLabel = DoForOneApproach(oriData,label,'PCA2','LOF')
    # KPCALOFLabel = DoForOneApproach(oriData,label,'KPCA','LOF')
    LOFLabel = DoForOneApproach(oriData,label,'to01','LOF')
    # plotShow(labelJudge(sampleRes,LOFLabel,100,68,0.1),labelJudge(sampleRes,AeLOFLabel,100,68,0.1))
    plotShow(labelJudge(sampleRes,LOFLabel,100,66,0.1),labelJudge(sampleRes,PCALOFLabel,100,66,0.1))
    plotShow(labelJudge(sampleRes,LOFLabel,100,66,0.1),labelJudge(sampleRes,PCA2LOFLabel,100,66,0.1))
    # plotShow(labelJudge(sampleRes,LOFLabel,100,66,0.1),labelJudge(sampleRes,KPCALOFLabel,100,66,0.1))

def plotShow(res,baseLine,type):
    plt.figure()
    x= range(1,100)
    l1, = plt.plot(x, baseLine.get(type), 'r-')
    l2, = plt.plot(x, res.get('AE').get(type), 'y-.')
    l3, = plt.plot(x, res.get('PCA').get(type), 'b--')
    l4, = plt.plot(x, res.get('KPCA').get(type), 'g*-')
    plt.legend(handles=[l1, l2, l3, l4 ], labels=['LOF' , 'AE-LOF', 'PCA-LOF' , 'KPCA-LOF'], loc='best')
    plt.show()



