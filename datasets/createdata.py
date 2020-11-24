# -*- coding: utf-8 -*- 
# @Time : 2020/11/24 2:27 下午 
# @Author : xwj
from termcolor import cprint
import datetime
import torch.utils.data.dataloader as dataloader
import math
import sys
import pdb
from termcolor import cprint
import torch
from matplotlib import cm
from tqdm import tqdm
import time
import shutil
import nibabel as nib
import pdb
import argparse
import matplotlib.pyplot as plt
import os
from torch.optim import lr_scheduler
import loguru
import io
import PIL.Image
import numpy as np
import glob
# -*- coding: utf-8 -*-
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
from tqdm import tqdm
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)

        if froze: self.frozePara()

    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps

    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False


class SumPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(SumPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        if type(kernel_size) is not int:
            self.area = kernel_size[0] * kernel_size[1]
        else:
            self.area = kernel_size * self.kernel_size

    def forward(self, dotmap):
        return self.avgpool(dotmap) * self.area


def gaussian_filter_density(img, points, boxes=None, beta=0.1):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

    return:
    density: the density-map we want. Same shape as input image but only has one channel.

    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''

    img_shape = [img.shape[0], img.shape[1]]
    num_faces = len(points)
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(num_faces), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(num_faces)
    if gt_count == 0:
        return density

    leafsize = 2048
    if gt_count > 1:
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * beta
        else:
            sigma = np.average(np.array([1, 1])) / 2. / 2.  # case: 1 point
        density_single = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        if boxes:
            mask = np.zeros(img.shape)
            mask[boxes[i][0]:boxes[i][2], boxes[i][1]:boxes[i][3]] = 1
            density_single = density_single * mask
            zoom = 1 / density_single.sum()
            density_single = density_single * zoom
        density = density + density_single

    return density


if __name__ == '__main__':
    f = open(r'../label.txt', 'r')
    lines = f.readlines()
    content = {}

    imgnames = []
    imgboxs = []

    for i in range(len(lines)):
        if lines[i][-2] == 'g':
            name = lines[i].strip()[2:]
            content[name] = {}
            content[name]['boxes'] = []
            content[name]['center'] = []
            content[name]['rawinfo'] = []

        else:
            content[name]['rawinfo'].append(lines[i].strip())
            label = [float(x) for x in lines[i].split(' ')]
            x1 = int(label[0])
            y1 = int(label[1])
            x2 = int(label[0] + label[2])
            y2 = int(label[1] + label[3])
            box_center = [x1, y1, x2, y2]
            content[name]['boxes'].append(box_center)
            content[name]['center'].append([(x1 + x2) / 2, (y1 + y2) / 2])

    f = open('data_state.pkl', 'wb')
    pickle.dump(content, f)
    f.close()

    for k, v in tqdm(content.items()):
        savename = '/widerface/density/' + k.replace('.jpg', '.pkl')
        img = cv2.imread(k)
        img_shape = [img.shape[0], img.shape[1]]
        density = gaussian_filter_density(img, v['center'], v['boxes'])
        f = open(savename, 'wb')
        pickle.dump(content, f)
        f.close()
