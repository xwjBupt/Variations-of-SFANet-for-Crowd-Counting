# -*- coding: utf-8 -*- 
# @Time : 2020/11/24 2:27 下午 
# @Author : xwj

import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb


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


def gaussian_filter_density(img, boxes=None, beta=0.1):
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
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(boxes), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(boxes)
    if gt_count == 0:
        return density
    for i, box in enumerate(boxes):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[int(center_y), int(center_x)] = 1.
        sig_w = w / 2
        sig_h = h / 2
        density_single = scipy.ndimage.filters.gaussian_filter(pt2d, [sig_h, sig_w], mode='constant')
        assert (abs(density_single.sum() - 1) < 0.5)
        zoom = 1 / density_single.sum()
        density_single *= zoom
        density += density_single
        # plt.suptitle(str(i))
        # plt.subplot(131)
        # plt.imshow(img)
        #
        # plt.subplot(132)
        # plt.title('single:%.3f' % density_single.sum())
        # plt.imshow(density_single * 100000, cmap=cm.jet)
        #
        # plt.subplot(133)
        # plt.title('all:%.3f' % density.sum())
        # plt.imshow(density * 100000, cmap=cm.jet)
        # # plt.show()
        # plt.pause(0.01)
        # print('#' * 10+str(i)+'#'*10)

    assert abs(density.sum() - gt_count) < 1.
    return density


if __name__ == '__main__':
    phase = 'train'
    f = open(r'/dfsdata2/xuwj16_data/widerface/' + phase + '/label.txt', 'r')
    lines = f.readlines()
    content = {}

    imgnames = []
    imgboxs = []

    for i in tqdm(range(len(lines))):
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
            if label[2] < 1:
                label[2] = 1
            if label[3] < 1:
                label[3] = 1
            x2 = int(label[0] + label[2])
            y2 = int(label[1] + label[3])
            box = [x1, y1, x2, y2]
            content[name]['boxes'].append(box)
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            content[name]['center'].append(center)

    f = open('/dfsdata2/xuwj16_data/widerface/' + phase + '/' + phase + '_data_state.pkl', 'wb')
    pickle.dump(content, f)
    f.close()

    root = '/dfsdata2/xuwj16_data/widerface/' + phase + '/images/'
    plt.ion()
    for k, v in tqdm(content.items()):
        savename = '/dfsdata2/xuwj16_data/widerface/' + phase + '/density/' + k.replace('.jpg', '.pkl')
        temp = {}
        img = cv2.imread(root + k)
        img_shape = [img.shape[0], img.shape[1]]
        density = gaussian_filter_density(img, v['boxes'])
        #pdb.set_trace()

        #         plt.subplot(121)
        #         plt.title('%.4f'%len(v['boxes']))
        #         plt.imshow(img)
        #         plt.subplot(122)
        #         plt.title('%.4f' % (density.sum()))
        #         plt.imshow(density * 100000)
        # plt.show()
        # plt.pause(0.05)
        # print(k)
        f = open(savename, 'wb')
        temp['data'] = density
        pickle.dump(temp, f)
        f.close()
        cv2.imwrite(savename.replace('.pkl','.png'),density*10000)

