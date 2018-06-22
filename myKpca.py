import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA

import jsonHandle
import returnResutl as rr
import myLof


def DoKPCA(kernel, pcaData, varN=None):
    # do pca
    print('Task KPCA : START TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    kpca = KernelPCA(varN , kernel=kernel, fit_inverse_transform=True)
    X_r = kpca.fit(pcaData).transform(pcaData)
    # print('explained variance ratio (first two components): %s' % str(kpca.explained_variance_ratio_))
    print(np.shape(X_r))
    print('Task KPCA : END TIME:'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return X_r

def dataEncode(kernel, oriData, varN=None):
    data = preprocessing.scale(oriData)
    res = DoKPCA(kernel, data, varN=varN)
    return res



def getDoKPCALOFIndex(kernel, oriData, per,varN=None):
    data = preprocessing.scale(oriData)

    res = DoKPCA(kernel, data, varN=varN)
    # doLof
    result = myLof.getTopIndex(res, per, need='false')
    return result


def test2(inst):
    data = jsonHandle.jsonDataRead('../no/' + inst + '/trainingListAvg1.json')
    print(np.shape(data))
    # data = preprocessing.scale(data)
    label = jsonHandle.jsonDataRead('../n/' + inst + '/labelReal.json')
    sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
    res1 = getDoKPCALOFIndex('poly', data, 1, varN= 30)
    res2 = getDoKPCALOFIndex('poly', data, 1, varN= 90)
    res11 = getDoKPCALOFIndex('rbf', data, 1, varN= 30)
    res21 = getDoKPCALOFIndex('rbf', data, 1, varN= 90)
    polyLabel1 = []
    polyLabel2 = []
    rbfLabel1 = []
    rbfLabel2 = []
    for i in range(len(res1)):
        polyLabel1.append(label[res1[i]])
        polyLabel2.append(label[res11[i]])
        rbfLabel1.append(label[res2[i]])
        rbfLabel2.append(label[res21[i]])

    polyR1 = rr.getResult(sampleRes, polyLabel1, 100, 205, 0.1)
    polyR11 = rr.getResult(sampleRes, polyLabel2, 100, 205, 0.1)
    rbfR1 = rr.getResult(sampleRes, rbfLabel1, 100, 205, 0.1)
    rbfR11 = rr.getResult(sampleRes, rbfLabel2, 100, 205, 0.1)
    plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2)
    l11, = ax1.plot(polyR1.get('recall'),polyR1.get('precision'), 'b-')
    l12, = ax1.plot(rbfR1.get('recall'),rbfR1.get('precision'), 'y-.')
    #    ax1.ylabel('precision')
    # ax1.set_xlabel('Top Percent Sample(%)')
    ax1.legend(handles=[l11, l12], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')
    ax1.set_title(" KPCA d'=30 ")  # 设置小图的标题

    ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=2)
    l21, = ax2.plot(polyR11.get('recall'),polyR11.get('precision'), 'b-')
    l22, = ax2.plot(rbfR11.get('recall'),rbfR11.get('precision'), 'y-.')
    #   ax2.ylabel('recall')
    ax2.set_xlabel('Top Percent Sample(%)')
    ax2.legend(handles=[l21, l22], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='lower right')
    ax2.set_title(" KPCA d'=90 ")  # 设置小图的标题
    # ax2.set_xlabel('Top Percent Sample(%)')
    ax1.set_xlabel('recall')
    ax2.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax2.set_ylabel('precision')
    plt.tight_layout()
    plt.show()


def test(inst):
    data = jsonHandle.jsonDataRead('../n/' + inst + '/trainingListAvg1.json')
    print(np.shape(data))
    # data = preprocessing.scale(data)
    label = jsonHandle.jsonDataRead('../n/' + inst + '/labelReal.json')
    sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")

    for i in range(10):

        res1 = getDoKPCALOFIndex('poly', data, 1, varN=(i*60)+30)
        res2 = getDoKPCALOFIndex('rbf', data, 1, varN=(i*60)*30)
        polyLabel = []
        rbfLabel = []
        for i in range(len(res1)):
            polyLabel.append(label[res1[i]])
            rbfLabel.append(label[res2[i]])
        polyR = rr.getResult(sampleRes, polyLabel, 100, 205, 0.1)
        rbfR = rr.getResult(sampleRes, rbfLabel, 100, 205, 0.1)
        plt.figure()

        x = range(polyR.get('slice'))
        l1, = plt.plot( polyR.get('recall'),polyR.get('precision'), 'b-')
        l2, = plt.plot( rbfR.get('recall'),rbfR.get('precision'), 'y-.')
        #    ax1.ylabel('precision')
        # ax1.set_xlabel('Top Percent Sample(%)')
        plt.legend(handles=[l1, l2], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')
        #plt.set_title('KPCA')  # 设置小图的标题


        # ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2)
        # x = range(polyR.get('slice'))
        # l1, = ax1.plot(x, polyR.get('precision'), 'b-')
        # l2, = ax1.plot(x, rbfR.get('precision'), 'y-.')
        # #    ax1.ylabel('precision')
        # # ax1.set_xlabel('Top Percent Sample(%)')
        # ax1.legend(handles=[l1, l2], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')
        # ax1.set_title('precision')  # 设置小图的标题
        #
        # ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=2)
        # l11, = ax2.plot(x, polyR.get('recall'), 'b-')
        # l21, = ax2.plot(x, rbfR.get('recall'), 'y-.')
        # #   ax2.ylabel('recall')
        # ax2.set_xlabel('Top Percent Sample(%)')
        # ax2.legend(handles=[l11, l21], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='lower right')
        # ax2.set_title('recall')  # 设置小图的标题
        # ax2.set_xlabel('Top Percent Sample(%)')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()

        # def test(inst):
        #     data = jsonHandle.jsonDataRead('../n/' + inst + '/trainingListAvg1.json')
        #     print(np.shape(data))
        #     # data = preprocessing.scale(data)
        #     label = jsonHandle.jsonDataRead('../n/' + inst + '/labelReal.json')
        #     sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
        #
        #     for i in range(10):
        #
        #         res1 = getDoKPCALOFIndex('poly', data, 1, varN=(i * 60) + 30)
        #         res2 = getDoKPCALOFIndex('rbf', data, 1, varN=(i * 60) * 30)
        #         polyLabel = []
        #         rbfLabel = []
        #         for i in range(len(res1)):
        #             polyLabel.append(label[res1[i]])
        #             rbfLabel.append(label[res2[i]])
        #         polyR = rr.getResult(sampleRes, polyLabel, 100, 205, 0.1)
        #         rbfR = rr.getResult(sampleRes, rbfLabel, 100, 205, 0.1)
        #         plt.figure()
        #
        #         x = range(polyR.get('slice'))
        #         l1, = plt.plot(polyR.get('recall'), polyR.get('precision'), 'b-')
        #         l2, = plt.plot(rbfR.get('recall'), rbfR.get('precision'), 'y-.')
        #         #    ax1.ylabel('precision')
        #         # ax1.set_xlabel('Top Percent Sample(%)')
        #         plt.legend(handles=[l1, l2], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')
        #         # plt.set_title('KPCA')  # 设置小图的标题
        #
        #
        #         # ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2)
        #         # x = range(polyR.get('slice'))
        #         # l1, = ax1.plot(x, polyR.get('precision'), 'b-')
        #         # l2, = ax1.plot(x, rbfR.get('precision'), 'y-.')
        #         # #    ax1.ylabel('precision')
        #         # # ax1.set_xlabel('Top Percent Sample(%)')
        #         # ax1.legend(handles=[l1, l2], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')
        #         # ax1.set_title('precision')  # 设置小图的标题
        #         #
        #         # ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=2)
        #         # l11, = ax2.plot(x, polyR.get('recall'), 'b-')
        #         # l21, = ax2.plot(x, rbfR.get('recall'), 'y-.')
        #         # #   ax2.ylabel('recall')
        #         # ax2.set_xlabel('Top Percent Sample(%)')
        #         # ax2.legend(handles=[l11, l21], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='lower right')
        #         # ax2.set_title('recall')  # 设置小图的标题
        #         # ax2.set_xlabel('Top Percent Sample(%)')
        #         plt.xlabel('recall')
        #         plt.ylabel('precision')
        #         plt.show()





#test2('s062901')