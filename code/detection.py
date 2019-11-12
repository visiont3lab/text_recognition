# -*- coding: utf-8 -*-
#python detection.py

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import torchvision
import matplotlib.pyplot as plt
import cv2

from craft import CRAFT
from collections import OrderedDict
from refinenet import RefineNet


class DETECTION:


    def __init__(self):
        # model settings #
        self.trained_model = 'model/craft_mlt_25k.pth'
        self.text_threshold=0.7
        self.low_text=0.4
        self.link_threshold=0.4
        self.cuda=True
        self.canvas_size=1280
        self.mag_ratio=1.5
        self.poly=True
        self.show_time=False
        self.video_folder='input/'
        self.refine=False
        self.refiner_model='model/craft_refiner_CTW1500.pth'
        self.interpolation = cv2.INTER_LINEAR
       
        #import model
        self.net = CRAFT() # initialize

    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def load_model(self):
        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model)))
        else:
            self.net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        # # LinkRefiner
        self.refine_net = None
        if self.refine:
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refiner_model + ')')
            if self.cuda:
                self.refine_net.load_state_dict(self.copyStateDict(torch.load(self.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(self.copyStateDict(torch.load(self.refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.poly = True
            t = time.time()

    def test_net(self,image_opencv):   
        
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image_opencv, self.canvas_size, interpolation=self.interpolation, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
       
        if self.cuda:
            x = x.cuda()
       
        # forward pass
        y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        t0 = time.time()
        if self.refine_net is not None:
            y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)
        #print(boxes)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        t1 = time.time() - t1

        if self.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
        return boxes, polys

    




