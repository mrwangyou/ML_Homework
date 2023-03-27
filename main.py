# import os

# sett = 'test'

# file_list = os.listdir('./sg_dataset/sg_{}_images'.format(sett))


# annotation = json.load(open('./json_dataset/annotations_{}.json'.format(sett)))

# print(annotation['10003466765_d20a7655c6_b.jpg'])

# for file in file_list:
#     try:
#         print(annotation[file][1])
#     except:
#         pass
#     else:
#         print(file)
#         break



import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

torch.set_num_threads(8)


# torch.manual_seed(4)
# random.seed()


class ImageDataset(Dataset):

    def __init__(self, file):

        if file == 'train':
            self.path = './sg_dataset/sg_train_images/'
            self.annotation = json.load(open('./json_dataset/annotations_train.json'))
        elif file == 'test':
            self.path = './sg_dataset/sg_test_images/'
            self.annotation = json.load(open('./json_dataset/annotations_test.json'))
        
        self.ImgList = os.listdir(self.path)
        # print(len(self.annotation))

    def __len__(self):

        return len(self.ImgList)

    def __getitem__(self, index):
        img = {}

        img['ID'] = self.ImgList[index]

        data = Image.open(self.path + img['ID'])
        w, h = data.size
        resize = transforms.Resize([128, 128])
        data = resize(data)
        img['data'] = torch.from_numpy(np.array(data)) / 256

        img['label'] = self.annotation[img['ID']]

        return img


class MyCNN(nn.Module):

    def __init__(self,
                 input_dim=3,
                 hidden_dim_1=128,
                 hidden_dim_2=64,
                 hidden_dim_3=32,
                 linear_dim=1,
                 kernel_size=9,
                 predicate_dim=70,
                 object_dim=100,
                 subject_dim=100,
                 batch_size=1,
                 ):
        super(MyCNN, self).__init__()
        self.batch_size = batch_size
        predicate_input = 115370
        object_input = 115371
        subject_input = 115371

        self.conv1 = nn.Conv2d(input_dim, hidden_dim_1, kernel_size)
        self.conv2 = nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size)
        self.conv3 = nn.Conv2d(hidden_dim_2, hidden_dim_3, kernel_size)
        self.linear1 = nn.Linear(predicate_input, predicate_dim)
        self.linear2 = nn.Linear(object_input, object_dim)
        self.linear3 = nn.Linear(subject_input, subject_dim)
        self.relu = nn.ReLU()

    def forward(self, img):  # 1*128*128*3
        output = torch.permute(img, (0, 3, 1, 2))  # 1*3*128*128
        
        output = self.conv1(output)  # 1*128*126*126
        output = self.conv2(output)  # 1*64*124*124
        output = self.conv3(output)  # 1*32*122*122
        # print(output.size())
        output = output.contiguous().view(self.batch_size, -1)

        predicate_output = output[:, :int(output.size(1) / 3)]
        object_output = output[:, int(output.size(1) / 3): int(output.size(1) / 3 * 2)]
        subject_output = output[:, int(output.size(1) / 3 * 2):]
        
        output1 = self.relu(self.linear1(predicate_output))
        output1 = F.log_softmax(output1, dim=1)

        output2 = self.relu(self.linear2(object_output))
        output2 = F.log_softmax(output2, dim=1)

        output3 = self.relu(self.linear3(subject_output))
        output3 = F.log_softmax(output3, dim=1)

        # output = torch.cat((output1, output2, output3), 1)

        return output1, output2, output3



# def get_accuracy(truth, pred):
#     assert len(truth) == len(pred)
#     right = 0
#     for i in range(len(truth)):
#         if truth[i] == pred[i]:
#             right += 1.0
#     return right / (len(truth) + 1)


def train():
    INPUT_DIM = 3
    HIDDEN_DIM_1 = 128
    HIDDEN_DIM_2 = 64
    HIDDEN_DIM_3 = 32
    LINEAR_DIM = 1
    KERNEL_SIZE = 9
    PREDICATE_DIM = 70
    OBJECT_DIM = 100
    SUBJECT_DIM = 100
    EPOCH = 100
    BATCH_SIZE = 1
    WEIGHT_DECAY = 1e-2
    SMALL_STEP_EPOCH = 10

    # img = Image.open('./sg_dataset/sg_train_images/1602315_961e6acf72_b.jpg')
    # img2 = torch.from_numpy(np.array(img))
    # print(img2.shape)
    
    trainLoader = DataLoader(dataset=ImageDataset('train'),
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             )
    # devLoader = DataLoader(dataset=ImageDataset('./listDev.txt'),
    #                        batch_size=BATCH_SIZE,
    #                        shuffle=True)
    testLoader = DataLoader(dataset=ImageDataset('test'),
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    model = MyCNN(
        input_dim=INPUT_DIM,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        hidden_dim_3=HIDDEN_DIM_3,
        linear_dim=LINEAR_DIM,
        kernel_size=KERNEL_SIZE,
        predicate_dim=PREDICATE_DIM,
        object_dim=OBJECT_DIM,
        subject_dim=SUBJECT_DIM,
        batch_size=BATCH_SIZE
    )

    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()

    loss_function = nn.NLLLoss()
    # loss_function = nn.MultiMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    bestTillNow = 0
    if 'bestModel' not in os.listdir('./'):
        os.mkdir('./bestModel')
    for i in range(EPOCH):
        print('epoch: {} start!'.format(i))
        train_epoch(model, trainLoader, loss_function, optimizer, i)
        # dev_acc = evaluate_epoch(model, devLoader, loss_function, i)
        test_acc = evaluate_epoch(model, testLoader, loss_function, i)
        torch.save(model.module.state_dict(), './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')
        # if i == SMALL_STEP_EPOCH:
        #     optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay = WEIGHT_DECAY)


def train_epoch(model,
                train_data,
                loss_function,
                optimizer,
                i,
                ):
    model.train()
    avg_loss = 0
    acc = 0
    cnt = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for batch in tqdm(train_data):
        cnt += 1
        # if cnt % 100 == 0:
        #     print('Begin ' + str(int(cnt / 100)) + ' batch in an epoch!')
        label = batch['label']
        data = batch['data']

        # label = label.cuda()
        # data = data.cuda()
        pred, obj, sub = model(data)
        try:
            if max(pred[0]) == pred[0, label[0]['predicate'].item()] and \
                max(obj[0]) == obj[0, label[0]['object']['category'].item()] and \
                max(sub[0]) == sub[0, label[0]['subject']['category'].item()]:
                acc = acc + 1
        except:
            pass

        try:
            loss_1 = loss_function(
                pred, label[0]['predicate'].cuda()
            )

            loss_2 = loss_function(
                obj, label[0]['object']['category'].cuda()
            )

            loss_3 = loss_function(
                sub, label[0]['subject']['category'].cuda()
            )

        

            loss = (loss_1 + loss_2 + loss_3) / 3

            avg_loss = avg_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    avg_loss = avg_loss / cnt
    acc = acc / float(cnt)
    print('epoch ' + str(i))
    print('train: ')
    print('loss: ' + str(avg_loss))
    print('acc: ' + str(acc))


def evaluate_epoch(model,
                   train_data,
                   loss_function,
                   i,
                   ):
    model.eval()
    avg_loss = 0
    acc = 0
    cnt = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for batch in tqdm(train_data):
        cnt += 1
        label = batch['label']
        data = batch['data']

        pred, obj, sub = model(data)
        flag = 0
        try:
            for key, _ in enumerate(label):
                if max(pred[0]) == pred[0, label[key]['predicate'].item()] and \
                    max(obj[0]) == obj[0, label[key]['object']['category'].item()] and \
                    max(sub[0]) == sub[0, label[key]['subject']['category'].item()]:
                    flag = 1
                    break
        except:
            pass
        
        if flag == 1:
            acc = acc + 1
            
        # try:
        #     loss_1 = loss_function(
        #         pred, label[0]['predicate'].cuda()
        #     )

        #     loss_2 = loss_function(
        #         obj, label[0]['object']['category'].cuda()
        #     )

        #     loss_3 = loss_function(
        #         sub, label[0]['subject']['category'].cuda()
        #     )

        

        #     loss = (loss_1 + loss_2 + loss_3) / 3

        #     avg_loss = avg_loss + loss

        # except:
        #     pass
    avg_loss = avg_loss / cnt
    acc = acc / float(cnt)
    print('test:')
    # print('loss: ' + str(avg_loss))
    print('acc: ' + str(acc))

    return acc

train()












