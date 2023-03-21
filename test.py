import os
import json
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# img = Image.open('./sg_dataset/sg_train_images/1602315_961e6acf72_b.jpg')

# resize = transforms.Resize([1024, 1024])
# img = resize(img)
# img = torch.from_numpy(np.array(img))
# img = img / 256

# conv = nn.Conv2d(1024, 8, [3, 3])
# conv2 = nn.Conv2d(1024, 8, [3, 3])

# tmp1 = conv(img)
# tmp2 = conv2(img)

# print(tmp1.size())
# print(tmp2.size())

# tmp3 = torch.cat((tmp1, tmp2), 0)

# # print(tmp3.size())

# tmp = torch.rand(2, 3)

# print(F.log_softmax(tmp, dim=1))




class Customer:
    def __init__(self):
        pass

    def __str__(self) -> str:
        return '123'


c = Customer()
print(c)















