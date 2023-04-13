import logging
import torch.nn as nn
from fastai.vision import *

from modules.model import _default_tfmer_cfg
from modules.model import Model
from modules.base_modules import *


class VSTModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.d_model = ifnone(config.model_d_model, 512)
        self.max_length = config.dataset_max_length + 1
        self.convnet = ConvNet(config)
        self.visual = Visual(config)
        self.vsalign = Vsalign(config)
        self.interaction = Interaction(config)
        self.cls1 = nn.Linear(self.d_model, self.charset.num_classes)
        self.cls2 = nn.Linear(self.d_model, self.charset.num_classes)
        if config.name == "vst-f":
            self.semantic = Semantic(config)
            self.cls3 = nn.Linear(2 * self.d_model, self.charset.num_classes)
        
        
    def forward(self, images, *args):
        """
        Args:
            images (bs, c, H, w)
        """
        cnn_features = self.convnet(images) #(bs, c, n)
        #将cnn_features转化为transformer接受的格式
        cnn_features = cnn_features.permute(2, 0, 1)
        visual1 = self.visual(cnn_features)
        semantic1 = self.vsalign(visual1)
        visual2, semantic2 = self.interaction(visual1, semantic1)
        semantic3 = self.vsalign(visual2)
        logits1 = self.cls1(semantic2.permute(1,0,2))
        ans1 = self.get_answer(semantic2, logits1, "language")
        logits2 = self.cls2(semantic3.permute(1,0,2))
        ans2 = self.get_answer(semantic3, logits2, "vision")
        if self.config.name == "vst-f":
            total = self.semantic(semantic2, semantic3)
            total = total.permute(1,0,2).reshape(-1, self.max_length, 2*self.d_model)
            logits3 = self.cls3(total)
            ans3 = self.get_answer(total, logits3, "alignment")
            return ans1, ans2, ans3
        return ans1, ans2
    
    def get_answer(self, features, logits, name):
        pt_lengths = self._get_length(logits)
        return {'feature': features, 'logits': logits, 'pt_lengths': pt_lengths,
                'loss_weight':1.0, 'name': name}
    


        