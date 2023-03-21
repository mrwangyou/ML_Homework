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
            self.annotation = json.load(open('./sg_dataset/sg_train_annotations.json'))
        elif file == 'test':
            self.path = './sg_dataset/sg_test_images/'
            self.annotation = json.load(open('./sg_dataset/sg_test_annotations.json'))
        
        self.ImgList = os.listdir(self.path)

    def __len__(self):

        return len(self.ImgList)

    def __getitem__(self, index):
        img = {}

        img['ID'] = self.ImgList[index]

        data = Image.open(self.path + img['ID'])
        w, h = data.size
        resize = transforms.Resize([2048, 2048])
        data = resize(data)
        img['data'] = torch.from_numpy(np.array(data))

        img['label'] = self.annotation[img['ID']][0]

        return img


class MyCNN(nn.Module):

    def __init__(self,
                 input_dim=2048,
                 hidden_dim=1024,
                 linear_dim=512,
                 predicate_dim=70,
                 object_dim=100,
                 subject_dim=100,
                 batch_size=1,
                 ):
        super(MyCNN, self).__init__()
        self.batch_size = batch_size
        self.linear_dim = linear_dim

        self.conv1 = nn.Conv2d(input_dim, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, linear_dim)
        assert linear_dim % 3 == 0
        self.linear1 = nn.Linear(linear_dim / 3, predicate_dim)
        self.linear2 = nn.Linear(linear_dim / 3, object_dim)
        self.linear3 = nn.Linear(linear_dim / 3, subject_dim)
        self.relu = nn.ReLU()

    def forward(self, img):  # 1*2048*2048*3
        output = self.conv1(img)  # 1*1024*1024*3
        output = self.conv2(output)  # 1*512*512*3
        output = output.view(self.batch_size, -1)
        predicate_output = output[:, :self.linear_dim / 3]
        object_output = output[:, self.linear_dim / 3: self.linear_dim / 3 * 2]
        subject_output = output[:, self.linear_dim / 3 * 2:]

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
    INPUT_DIM = 2048
    HIDDEN_DIM = 1024
    LINEAR_DIM = 512
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
        hidden_dim=HIDDEN_DIM,
        linear_dim=LINEAR_DIM,
        predicate_dim=PREDICATE_DIM,
        object_dim=OBJECT_DIM,
        subject_dim=SUBJECT_DIM,
        batch_size=BATCH_SIZE
    )

    device_ids = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()

    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    bestTillNow = 0
    for i in range(EPOCH):
        print('epoch: {} start!'.format(i))
        train_epoch(model, trainLoader, loss_function, optimizer, i)
        # dev_acc = evaluate_epoch(model, devLoader, loss_function, i, 1)
        test_acc = evaluate_epoch(model, testLoader, loss_function, i, 2)
        torch.save(model.module.state_dict(), './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')
        if i == SMALL_STEP_EPOCH:
            optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay = WEIGHT_DECAY)


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

    for batch in train_data:
        cnt += 1
        # if cnt % 100 == 0:
        #     print('Begin ' + str(int(cnt / 100)) + ' batch in an epoch!')
        label = batch['label']
        sent = batch['tweet'].squeeze(0)
        property = batch['property']
        label = label.cuda()
        sent = sent.cuda()
        property = property.cuda()
        pred = model(sent, property).unsqueeze(0)
        if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][1] and label == 1:
            acc = acc + 1
        if pred[0][0] > pred[0][1] and label == 0:
            TP = TP + 1
        if pred[0][0] > pred[0][1] and label == 1:
            FP = FP + 1
        if pred[0][0] < pred[0][1] and label == 1:
            TN = TN + 1
        if pred[0][0] < pred[0][1] and label == 0:
            FN = FN + 1

        loss = loss_function(pred, label)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = avg_loss / cnt
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = acc / float(cnt)
    specificity = TN / (TN + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('epoch ' + str(i))
    print('train: ')
    print('loss: ' + str(avg_loss))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('acc: ' + str(acc))
    print('specificity: ' + str(specificity))
    print('F1: ' + str(F1))
    print('MCC: ' + str(MCC))


def evaluate_epoch(model,
                   train_data,
                   loss_function,
                   i,
                   ii):
    model.eval()
    avg_loss = 0
    acc = 0
    cnt = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for batch in train_data:
        cnt = cnt + 1
        label = batch['label']
        sent = batch['tweet'].squeeze(0)
        property = batch['property']
        label = label.cuda()
        sent = sent.cuda()
        property = property.cuda()
        pred = model(sent, property).unsqueeze(0)
        if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][
            1] and label == 1:  # 0 for bot and 1 for human
            acc = acc + 1
        if pred[0][0] > pred[0][1] and label == 0:
            TP = TP + 1
        if pred[0][0] > pred[0][1] and label == 1:
            FP = FP + 1
        if pred[0][0] < pred[0][1] and label == 1:
            TN = TN + 1
        if pred[0][0] < pred[0][1] and label == 0:
            FN = FN + 1
        model.zero_grad()

        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss = avg_loss / cnt
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = acc / float(cnt)
    specificity = TN / (TN + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if ii == 1:
        print('val: ')
    else:
        print('test: ')
    print('loss: ' + str(avg_loss))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('acc: ' + str(acc))
    print('specificity: ' + str(specificity))
    print('F1: ' + str(F1))
    print('MCC: ' + str(MCC))

    return acc


# embed_mat = torch.load('./embedding.pt')
train()












