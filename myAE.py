import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from torch.autograd import Variable
import processMain as pm
import jsonHandle


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(331, 150),
            nn.Sigmoid()

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
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )
        self.softmax = nn.Sequential(
            # nn.Linear(3, 16),
            # nn.Tanh(),
            # nn.Linear(16, 64),
            # nn.Tanh(),
            nn.Linear(150, 2),
            nn.Sigmoid(),       # 激励函数让输出值在 (0, 1)
        )



    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def softmax(self, x):
        encoded = self.encoder(x)
        label = self.softmax(encoded)
        return label

def logisticRegression(data, EPOCH = 1000, BATCH_SIZE = 32, LR = 0.0005):
    return 1




def dataEncode(data, EPOCH = 1000, BATCH_SIZE = 32, LR = 0.0005):
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

def myAEMain(inst):
    oriData = jsonHandle.jsonDataRead('no/' + inst + '/trainingListAvg1.json')
    label = jsonHandle.jsonDataRead('no/' + inst + '/labelReal.json')
    # LOFLabel = pm.DoForOneApproach(oriData, label, 'to01', 'LOF')
    # sampleRes = jsonHandle.jsonDataRead("labelReal062901.json")
    #for i in range(100)
    aee =dataEncode(oriData)
    # aeRes = pm.labelJudge(sampleRes, pm.DoForOneApproach(oriData, label, 'AE', 'LOF'),100,205,0.1)
    np.save('aeEncode.npy', aee.data.numpy())
    # pm.plotShow(pm.labelJudge(sampleRes,LOFLabel,100,205,0.1),aeRes)


    # jsonHandle.jsonWrite("encoded_data.json", output)
    # X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    # fig = plt.figure(2);
    # ax = Axes3D(fig)
    # for x, y, z in zip(X, Y, Z ):
    #     ax.text(x, y, z ,'*')
    # ax.set_xlim(X.min(), X.max());
    # ax.set_ylim(Y.min(), Y.max());
    # ax.set_zlim(Z.min(), Z.max())
    # plt.show()

# myAEMain("s062901")