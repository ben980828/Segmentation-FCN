import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import imageio
import scipy.misc
import argparse
import glob
import os
import sys
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.vgg = models.vgg16(pretrained=True)
        features = self.vgg.features
        features[0].padding = (100, 100)
        features_pool3 = features[:17]
        features_pool4 = features[17:24]
        features_rest = features[24:]
        # self.features_extract = nn.Sequential(*features)
        self.features_extract1 = nn.Sequential(*features_pool3)
        self.features_extract2 = nn.Sequential(*features_pool4)
        self.features_extract3 = nn.Sequential(*features_rest)


        conv1 = nn.Conv2d(512, 4096, 7)
        torch.nn.init.kaiming_uniform_(conv1.weight, nonlinearity='relu')
        conv2 = nn.Conv2d(4096, 4096, 1)
        torch.nn.init.kaiming_uniform_(conv2.weight, nonlinearity='relu')

        self.conv_block = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.Dropout2d(), 
            conv2,
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        convscore = nn.Conv2d(4096, 7, kernel_size=1)
        torch.nn.init.kaiming_uniform_(convscore.weight, nonlinearity='relu')
        self.conv_score = convscore

        pool3score = nn.Conv2d(256, 7, kernel_size=1)
        torch.nn.init.kaiming_uniform_(pool3score.weight, nonlinearity='relu')
        self.pool3_score = pool3score

        pool4score = nn.Conv2d(512, 7, kernel_size=1)
        torch.nn.init.kaiming_uniform_(pool4score.weight, nonlinearity='relu')
        self.pool4_score = pool4score

        self.convT1 = nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=4, stride=2, bias=False)
        # 34x34

        self.convT2 = nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=4, stride=2, bias=False)

        self.convT3 = nn.ConvTranspose2d(in_channels=7, out_channels=7, kernel_size=16, stride=8, bias=False)

    def forward(self, input):
        x_pool3 = self.features_extract1(input)
        # 256 channel
        x_pool4 = self.features_extract2(x_pool3)
        # 512 channel
        x_out = self.features_extract3(x_pool4)

        x_conv = self.conv_block(x_out)
        x_score = self.conv_score(x_conv)
        x_convT1 = self.convT1(x_score)
        upscore2 = x_convT1# 34x34

        pool4_score = self.pool4_score(0.01*x_pool4)
        pool4 = pool4_score[:, :, 5:(5 + upscore2.size()[2]), 5:(5 + upscore2.size()[3])]
        # 7 channel

        pool4_convT1 = upscore2 + pool4
        upscore8 = self.convT2(pool4_convT1)

        pool3_score = self.pool3_score(0.0001*x_pool3)
        pool3 = pool3_score[:, :, 9:(9 + upscore8.size()[2]), 9:(9 + upscore8.size()[3])]

        sum = upscore8 + pool3
        upscore_sum = self.convT3(sum)
        x_output = upscore_sum[:, :, 28:(28 + input.size()[2]), 28:(28 + input.size()[3])].contiguous()

        return x_output

def output_to_mask(output):
    color_mask = np.zeros((output.shape[0]*output.shape[1], 3))
    loc_0 = np.where(output.flatten() == 0, 1, 0)
    color_mask[loc_0 == 1, :] = np.array([0, 1, 1])
    loc_1 = np.where(output.flatten() == 1, 1, 0)
    color_mask[loc_1 == 1, :] = np.array([1, 1, 0])
    loc_2 = np.where(output.flatten() == 2, 1, 0)
    color_mask[loc_2 == 1, :] = np.array([1, 0, 1])
    loc_3 = np.where(output.flatten() == 3, 1, 0)
    color_mask[loc_3 == 1, :] = np.array([0, 1, 0])
    loc_4 = np.where(output.flatten() == 4, 1, 0)
    color_mask[loc_4 == 1, :] = np.array([0, 0, 1])
    loc_5 = np.where(output.flatten() == 5, 1, 0)
    color_mask[loc_5 == 1, :] = np.array([1, 1, 1])
    loc_6 = np.where(output.flatten() == 6, 1, 0)
    color_mask[loc_6 == 1, :] = np.array([0, 0, 0])
    mask = np.reshape(color_mask, (output.shape[0], output.shape[1], 3))
    return mask

pyfile = sys.argv[0]
input_folder = sys.argv[1]
output_folder = sys.argv[2]

# input_folder = 'hw2-ben980828/hw2_data/p2_data/validation'
# output_folder = 'testfcn8'
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

model = Net()
state = torch.load('fcn8s_optimal_acc0.711498.pth')
# print(model._modules.keys())
model.load_state_dict(state)
# print(model)
model.to(device)
model.eval()#must remember


sat_image = []
all_image_path = os.listdir(input_folder)
for fn in all_image_path:
    if fn.endswith('sat.jpg'):
        sat_image.append(fn)
# print(sat_image)

for filename in sat_image:
    abs_path = os.path.join(input_folder, filename)
    image = Image.open(abs_path).convert('RGB')
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]
          )
    image = transform(image)
    image_ch = torch.unsqueeze(image, 0)
    model_input = image_ch.to(device)
    output = model(model_input)
    output_pred = torch.argmax(output, dim=1).int().cpu().detach().numpy()#1x512x512
    pred = np.squeeze(output_pred)# 512x512
    # print(pred)
    pred_mask = output_to_mask(pred)*255
    pred_mask = np.uint8(pred_mask)
    pred_mask = Image.fromarray(pred_mask)
    # print(pred_mask.shape) 512x512x3
    final_image = pred_mask.save(os.path.join(output_folder, filename[:4]+'_mask.png'), format='PNG')