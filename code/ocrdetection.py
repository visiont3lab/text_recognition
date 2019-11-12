#
#pip install lmdb
#pip install fire
#pip install natsort


# CUDA_VISIBLE_DEVICES=0 python _demo.py

import cv2
import string
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

class OCR:

    def __init__(self):
        
        # model settings #
        self.path_model= 'models/TPS-ResNet-BiLSTM-Attn.pth'
        self.batch_size = 1
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.Transformation = 'TPS'
        self.FeatureExtraction ='ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        parser = argparse.ArgumentParser()
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        self.opt = parser.parse_args()

        self.opt.num_gpu = torch.cuda.device_count()

        # load model
        self.converter = AttnLabelConverter(self.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)
        self.model = torch.nn.DataParallel(self.model).to('cuda:0')

        # load model
        self.model.load_state_dict(torch.load(self.path_model))


    def run(self,img):
        with torch.no_grad():
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = cv2.resize(img, dsize=(100, 32), interpolation=cv2.INTER_CUBIC)
            image_tensor = img[np.newaxis,np.newaxis, ...]

            image = torch.from_numpy(image_tensor).float().to(self.device)
            # For max length prediction
            length_for_pred = torch.IntTensor([self.batch_max_length] * self.batch_size).to(self.device)
            text_for_pred = torch.LongTensor(self.batch_size, self.batch_max_length + 1).fill_(0).to(self.device)
         
            preds = self.model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

            pred = ''
            for i in range(len(preds_str)):
                pred += preds_str[i][:preds_str[i].find('[s]')]  # prune after "end of sentence" token ([s])
            print(pred)
            
            return pred



