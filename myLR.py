import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
import datetime


def DoLR(train, classLabel, test):
    logreg = linear_model.LogisticRegression( tol=1e-6)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train, classLabel)
    print('------start------')
    t1 = datetime.datetime.now().microsecond
    Z = logreg.predict(test)
    t2 = datetime.datetime.now().microsecond
    print(t2-t1)
    print('------end------')
    return Z

# 定义超参数
batch_size = 32
learning_rate = 1e-3
num_epoches = 100






def DoLogisticR():
    return 1