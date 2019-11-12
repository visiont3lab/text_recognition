import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

imgW = 100
imgH = 32
batch_max_length = 25
Transformation = 'TPS'
FeatureExtraction ='ResNet'
SequenceModeling = 'BiLSTM'
Prediction = 'Attn'
num_fiducial = 20
input_channel = 1
output_channel = 512
hidden_size = 256


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': Transformation, 'Feat': FeatureExtraction,
                       'Seq': SequenceModeling, 'Pred': Prediction}

        """ Transformation """
        if Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=batch_max_length)

        return prediction
