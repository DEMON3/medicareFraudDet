import math

import matplotlib.pyplot as plt
import numpy as np

import jsonHandle


def getResult(sampleResult, realResult, slice, falseNum,per):

    dat = set(sampleResult);
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
        if (dat.__contains__(realResult[i])):
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
            cf = 2*sumCount/((i+1)+falseNum);
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


def resultProcess(fileName):
    realResult = jsonHandle.jsonDataRead(fileName)
    trueResult = []
    tempSet = set()
    for i in range(len(realResult)):
        cur = realResult[i].split(',')[0]
        if(not tempSet.__contains__(cur)):
            tempSet.add(cur)
            trueResult.append(cur)
        else:
            continue
    return trueResult





def main():
    lofRes = resultProcess('res062901/mLOFRes.json')
    pcaLOFRes = resultProcess('res062901/mpcaLOFRes.json')
    pcaPolyRes = resultProcess('res062901/tpcaPolyRes.json')
    pcaRbfRes = resultProcess('res062901/mpcaRbfRes.json')
    pcaRes = resultProcess('res062901/mpcaRes.json')

    sampleRes= jsonHandle.jsonDataRead("labelReal062901.json")
    print('lof')
    lof = getResult(sampleRes,lofRes,100,205,0.1)
    print('pcalof')

    pcalof = getResult(sampleRes,pcaLOFRes,100,205,0.1)
    print('pcaRbflof')
    pcaPolylof = getResult(sampleRes,pcaPolyRes,100,205,0.1)
    pcaRbflof = getResult(sampleRes,pcaRbfRes,100,205,0.1)
    print('pcaR')
    pca = getResult(sampleRes,pcaRes,100,205,0.1)

    plt.figure()
    l1, = plt.plot(lof.get('recall'), lof.get('precision'), 'r-')
    l2, = plt.plot(pcalof.get('recall'), pcalof.get('precision'), 'y-.')
    #l3, = plt.plot(pca.get('recall'), pca.get('precision'), 'b--')
    #l4, = plt.plot(pcaPolylof.get('recall'), pcaPolylof.get('precision'), 'g:')
    l4, = plt.plot(pcaRbflof.get('recall'), pcaRbflof.get('precision'), 'g:')
    # l1, = plt.plot(lof.get('fpr'), lof.get('recall'), 'r-')
    # l2, = plt.plot(pcalof.get('fpr'), pcalof.get('recall'), 'y-.')
    # l3, = plt.plot(pca.get('fpr'), pca.get('recall'), 'b--')
    # l4, = plt.plot(pcaRbflof.get('fpr'), pcaRbflof.get('recall'), 'g:')
    plt.legend(handles=[l1, l2, l4], labels=['LOF', 'PCA-LOF', 'rbfKPCA-LOF'], loc='best')
    # ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2)
    #
    # x = range(lof.get('slice'))
    # l1, = ax1.plot(x, lof.get('precision'), 'r-')
    # l2, = ax1.plot(x, pcalof.get('precision'),'y-.')
    # l3, = ax1.plot(x, pca.get('precision'), 'b--')
    # # l4, = ax1.plot(x, pcaPolylof.get('precision'), color='orange')
    # #    ax1.ylabel('precision')
    # # ax1.set_xlabel('Top Percent Sample(%)')
    # ax1.legend(handles=[l1, l2, l3], labels=['LOF', 'PCA-LOF', 'PCAR'], loc='best')
    # ax1.set_title('precision')  # 设置小图的标题
    #
    # ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=2)
    # l11, = ax2.plot(x, lof.get('recall'), 'r-')
    # l21, = ax2.plot(x, pcalof.get('recall'), 'y-.')
    # l31, = ax2.plot(x, pca.get('recall'), 'b--')
    # # l41, = ax2.plot(x, pcaPolylof.get('recall'), color='orange')
    # #   ax2.ylabel('recall')
    # ax2.set_xlabel('Top Percent Sample(%)')
    # ax2.legend(handles=[l11, l21, l31], labels=['LOF', 'PCA-LOF', 'PCAR'], loc='lower right')
    # ax2.set_title('recall')  # 设置小图的标题
    # ax2.set_xlabel('Top Percent Sample(%)')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

# def pltOne(x):
#     plt.figure()
#     l1, = plt.plot(x.get('recall'), x.get('precision'), 'b-')
#     l2, = plt.plot(y.get('recall'), y.get('precision'), 'y-.')
#     #    ax1.ylabel('precision')
#     # ax1.set_xlabel('Top Percent Sample(%)')
#     plt.legend(handles=[l1, l2], labels=['polyKPCALOF', 'rbfKPCALOF'], loc='best')

def compareWithStat():
    lofRes1 = resultProcess('res062901/LOFRes.json')
    lofRes = resultProcess('res062901/tLOFRes.json')
    statlofRes = resultProcess('res062901/statRes.json')

    sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")

    lof = getResult(sampleRes, lofRes, 100, 205, 0.1)
    lof1 = getResult(sampleRes, lofRes1, 100, 205, 0.1)
    statlof = getResult(sampleRes, statlofRes, 100, 205, 0.1)

    plt.figure()
    l1, = plt.plot(lof.get('recall'), lof.get('precision'), 'b-')
    l2, = plt.plot(statlof.get('recall'), statlof.get('precision'), 'y-.')
    l3, = plt.plot(lof1.get('recall'), lof1.get('precision'), 'r--')
    plt.legend(handles=[l1, l2], labels=['oriLOF', 'statLOF'], loc='best')
    plt.xlabel('recall')
    plt.ylabel('precision')
    # ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=2)
    #
    # x = range(lof.get('slice'))
    # l1, = ax1.plot(x, lof.get('precision'), 'b-')
    # l2, = ax1.plot(x, statlof.get('precision'), 'y-.')
    # #    ax1.ylabel('precision')
    # # ax1.set_xlabel('Top Percent Sample(%)')
    # ax1.legend(handles=[l1, l2], labels=['LOF', 'statLOF'], loc='best')
    # ax1.set_title('precision')  # 设置小图的标题
    #
    # ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=2)
    # l11, = ax2.plot(x, lof.get('recall'), 'b-')
    # l21, = ax2.plot(x, statlof.get('recall'), 'y-.')
    # #   ax2.ylabel('recall')
    # ax2.set_xlabel('Top Percent Sample(%)')
    # ax2.legend(handles=[l11, l21], labels=['LOF', 'statLOF'], loc='lower right')
    # ax2.set_title('recall')  # 设置小图的标题
    # ax2.set_xlabel('Top Percent Sample(%)')
    plt.show()


#compareWithStat()
#main()

