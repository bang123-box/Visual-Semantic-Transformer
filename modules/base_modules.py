import logging
import torch.nn as nn
from fastai.vision import *

from modules.resnet import resnet
from modules.transformer import (PositionalEncoding, 
                                 TransformerEncoder,
                                 TransformerEncoderLayer)

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_channels = ifnone(config.model_d_model, 512)
        blocks= ifnone(config.model_convnet_blocks, [1,1,2,5,3]) 
        striders=ifnone(config.model_convnet_striders, [(1,1), (2,2), (1,1), (2,2), (2, 1)])
        self.backbone = resnet(blocks, striders)

    def forward(self, images, *args):
        features = self.backbone(images)
        b, c, h, w = features.shape 
        features = features.reshape(b, c, -1)
        return features

class Vsalign(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_channels = ifnone(config.model_d_model, 512)
        self.l = config.dataset_max_length + 1
        self.q = nn.Parameter(torch.rand((self.l, self.out_channels)))
    
    def forward(self, input):
        #input (n, bs, c)
        input = input.permute(1, 0, 2) #(bs, n, c)
        atten_score = input @ self.q.T# (bs, n, t)
        atten_score = atten_score.permute(0,2,1).softmax(-1) #(bs, t, n)
        output = torch.bmm(atten_score, input) #(bs, t, c)
        return output.permute(1, 0, 2)

class Visual(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = ifnone(config.model_d_model, 512)
        nhead = ifnone(config.model_visual_nhead, 8)
        d_inner = ifnone(config.model_visual_d_inner, 2048)
        dropout = ifnone(config.model_visual_dropout, 0.1)
        activation = ifnone(config.model_visual_activation, "relu")
        num_layers = ifnone(config.model_visual_num_layers, 3)
        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=6*40)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_inner, dropout, 
                activation)
        norm = nn.LayerNorm(d_model)
        self.model = TransformerEncoder(encoder_layer, num_layers, norm)

    def forward(self, image_features):
        #image_features (n, bs, c)
        image_features = self.pos_encoder(image_features)
        visual_feature = self.model(image_features)
        return visual_feature

class Interaction(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = ifnone(config.model_d_model, 512)
        nhead = ifnone(config.model_interaction_nhead, 8)
        d_inner = ifnone(config.model_interaction_d_inner, 2048)
        dropout = ifnone(config.model_interaction_dropout, 0.1)
        activation = ifnone(config.model_interactionl_activation, "relu")
        num_layers = ifnone(config.model_interaction_num_layers, 3)
        self.d_model = d_model
        self.max_length = config.dataset_max_length + 1  # additional stop token
        
        self.pos_visual_encoder = PositionalEncoding(d_model, dropout=0, max_len=6*40)
        self.pos_semantic_encoder = nn.Parameter(torch.rand((self.max_length, 1, d_model)))
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_inner, dropout, 
                activation)
        norm = nn.LayerNorm(d_model)
        self.model = TransformerEncoder(encoder_layer, num_layers, norm)

    def forward(self, visual1, semantic1):
        #visual (n, bs, c), semantic(t, bs, c)
        visual1 = self.pos_visual_encoder(visual1)
        semantic1 = semantic1 + self.pos_semantic_encoder[:semantic1.shape[0], :]

        total = torch.cat([visual1, semantic1], 0)
        total = self.model(total)
        visual2, semantic2 = total[:6*40], total[6*40:]
        return visual2, semantic2


class Semantic(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = ifnone(config.model_d_model, 512)
        nhead = ifnone(config.model_semantic_nhead, 8)
        d_inner = ifnone(config.model_semantic_d_inner, 2048)
        dropout = ifnone(config.model_semantic_dropout, 0.1)
        activation = ifnone(config.model_semantic_activation, "relu")
        num_layers = ifnone(config.model_semantic_num_layers, 3)
        self.d_model = d_model
        self.max_length = config.dataset_max_length + 1  # additional stop token
        
        self.pos_visual_encoder = PositionalEncoding(d_model, dropout=0, max_len=2*self.max_length)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_inner, dropout, 
                activation)
        norm = nn.LayerNorm(d_model)
        self.model = TransformerEncoder(encoder_layer, num_layers, norm)

    def forward(self, semantic2, semantic3):
        #senantic(t, bs, c), semantic3(t, bs, c)
        total = torch.cat([semantic2, semantic3], 0)
        total = self.pos_visual_encoder(total)
        total = self.model(total)
        #(2t, bs, c)
        return total      