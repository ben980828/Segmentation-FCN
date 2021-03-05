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
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
from collections import OrderedDict

class Sat_Image(Dataset):
    def __init__(self, fileroot, image_root, mask_root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform

        # read filenames
        self.len = len(self.image_root)  
    def my_transform(self, image, mask):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.randrange(0, 271, 90)
            image.rotate(angle)
            mask.rotate(angle)
            
        return image, mask     

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.image_root[index]
        image = Image.open(image_fn).convert('RGB')    
        mask_fn = self.mask_root[index]
        mask = Image.open(mask_fn).convert('RGB')

        image, mask = self.my_transform(image, mask)
        image = self.transform(image)

        masks = np.empty((512, 512))
        mask = np.array(mask)
        # mask = self.transform(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        masks[mask == 4] = 6  # (Red: 100) Unknown
        masks = torch.tensor(masks, dtype=torch.long)   

        return image, masks

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
class Valid_Image(Dataset):
    def __init__(self, fileroot, image_root, mask_root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform

        # read filenames
        self.len = len(self.image_root)       
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.image_root[index]
        image = Image.open(image_fn).convert('RGB')
        image = self.transform(image)

        mask_fn = self.mask_root[index]
        mask = Image.open(mask_fn).convert('RGB')
        masks = np.empty((512, 512))
        mask = np.array(mask)
        # mask = self.transform(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        masks[mask == 4] = 6  # (Red: 100) Unknown
        masks = torch.tensor(masks, dtype=torch.long)   

        return image, masks

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def FCN_block(in_channels,out_channels,kernal):
    
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernal, stride=1)
    # nn.init.xavier_uniform_(conv_layer.weight)

    fcn_block = nn.Sequential(
        conv_layer,
        nn.ReLU(inplace=True),
        nn.Dropout()               
    )

    return fcn_block


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


def main():
    train_root = 'hw2-ben980828/hw2_data/p2_data/train/'
    valid_root = 'hw2-ben980828/hw2_data/p2_data/validation/'
    train_img = []
    train_mask = []
    val_img = []
    val_mask = []

    sat_image = glob.glob(os.path.join(train_root, '*_'+'sat.jpg'))
    for fn in sat_image:
        train_img.append(fn)
    sat_mask = glob.glob(os.path.join(train_root, '*_'+'mask.png'))
    for fn_mask in sat_mask:
        train_mask.append(fn_mask)

    val_sat_image = glob.glob(os.path.join(valid_root, '*_'+'sat.jpg'))
    for val_fn in val_sat_image:
        val_img.append(val_fn)
    val_sat_mask = glob.glob(os.path.join(valid_root, '*_'+'mask.png'))
    for val_mask_fn in val_sat_mask:
        val_mask.append(val_mask_fn)


    train_set = Sat_Image(fileroot=train_root, 
        image_root=train_img, 
        mask_root=train_mask,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]
          )
        )
    validation_set = Valid_Image(fileroot=valid_root,
        image_root=val_img, 
        mask_root=val_mask,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]
          )
        )
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=1)

    model = Net()
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max')
    criterion = nn.CrossEntropyLoss()

    epoch = 30
    small_value = 1e-6
    max_iou = 0.
    iteration = 0
    log_interval = 100

    # training
    for ep in range(0, epoch):
        model.train()
        train_iou = 0.
        train_data_num = 0
        train_bar = tqdm(total=len(train_loader))
        print('Current training epoch : ', ep)
        for data, target in train_loader:
            batchsize = data.size(0)
            train_data_num += batchsize

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(output, dim=1).cpu().detach().numpy()
            b = target.cpu().detach().numpy()
            for batch in range(batchsize):
                mean_iou = 0
                for i in range(6):
                    tp_fp = np.sum(pred[batch] == i)
                    tp_fn = np.sum(b[batch] == i)
                    tp = np.sum((pred[batch] == i) * (b[batch] == i))
                    iou = (tp + small_value) / (tp_fp + tp_fn - tp + small_value)
                    mean_iou += iou / 6
                train_iou += mean_iou


            postfix = OrderedDict([
            ('train_acc', train_iou/train_data_num),
            ])
            train_bar.set_postfix(postfix)
            train_bar.update(1)
            # if iteration % log_interval == 0:
            #     print('Train Epoch: {} \tLoss: {:.6f}'.format(
            #         epoch, loss.item()))
            #     print('Acc = {}'.format(train_iou/train_data_num))
            iteration += 1
        train_bar.close()

        # validation
        model.eval()

        val_loss = 0
        acc = 0
        valid_data_num = 0
        val_batchsize = 0
        with torch.no_grad():
            valid_bar = tqdm(total=len(validation_loader))
            for data, target in validation_loader:
                val_batchsize = data.size(0)
                valid_data_num += val_batchsize
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target)
                pred = torch.argmax(output, dim=1).cpu().detach().numpy()
                b = target.cpu().detach().numpy()
                for batch in range(val_batchsize):
                    mean_iou = 0
                    for i in range(6):
                        tp_fp = np.sum(pred[batch] == i)
                        tp_fn = np.sum(b[batch] == i)
                        tp = np.sum((pred[batch] == i) * (b[batch] == i))
                        iou = (tp + small_value) / (tp_fp + tp_fn - tp + small_value)
                        mean_iou += iou / 6
                    acc += mean_iou


                valid_postfix = OrderedDict([
                  ('valid_acc', acc/valid_data_num),
                ])
                valid_bar.set_postfix(valid_postfix)
                valid_bar.update(1)
            valid_bar.close()

        val_loss /= len(validation_loader.dataset)
        acc /= valid_data_num
        print('\nTest set: Average loss: {:.4f}'.format(val_loss))
        print('Accuracy = {}\n'.format(acc))

        if acc > max_iou:
            print('Performance improved : ({:.3f} --> {:.3f}). Save model ==> '.format(max_iou, acc))
            max_iou = acc
            torch.save(model.state_dict(), 'fcn8s_optimal_acc{}.pth'.format(acc))
        # lr_decay.step(acc)
    print('Final max acc : ', max_iou)

if __name__ == '__main__':
    main()
