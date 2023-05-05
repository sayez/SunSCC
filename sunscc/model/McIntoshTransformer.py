from operator import indexOf
from re import I
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T

from bioblue.model.TransUNet import *

import matplotlib.pyplot as plt

class McIntoshTransformerClassifier(nn.Module):
    def __init__(self, 
                input_format,
                output_format,
                
                classes, 

                first_classes,
                second_classes,
                third_classes,
                
                architecture=None,
                focus_on_group=False,
                focus_on_largest_sunspot=False,
                ):
        super().__init__()


        self.__dict__.update(locals())
        
        self.input_format = input_format
        self.output_format = output_format
        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        self.focus_on_group = focus_on_group
        self.focus_on_largest_sunspot = focus_on_largest_sunspot
        
        
        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        self.img_dim = self.architecture.encoder['img_dim'] 
        self.in_channels = self.architecture.encoder['in_channels'] 
        self.out_channels = self.architecture.encoder['out_channels'] 
        self.head_num = self.architecture.encoder['head_num'] 
        self.mlp_dim = self.architecture.encoder['mlp_dim'] 
        self.block_num = self.architecture.encoder['block_num'] 
        self.patch_dim = self.architecture.encoder['patch_dim'] 
        self.embedding_dim = self.architecture.encoder['embedding_dim']
        self.class_num = len(classes) + 1

        self.encoder = Encoder(self.img_dim, self.in_channels, self.out_channels,
                               self.head_num, self.mlp_dim, self.block_num, self.patch_dim,self.embedding_dim)

        self.globalAvgPooling =  nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1))
        )
       
        in_feats = self.embedding_dim
        self.fc_input_size = in_feats + (len(self.input_format)-2) # concat inputs that are not 'image' and 'mask'
        self.fc2_input_size = self.fc_input_size
        self.fc3_input_size = self.fc_input_size

        #  Create 3 classifiers (1 per McIntosh letter), each taking the encoded image as input.
        lst = [nn.Linear(self.mp1_hidden_widths[i], self.mp1_hidden_widths[i+1]) for i in range(len(self.mp1_hidden_widths)-1) ]
        lst2 = [nn.ReLU()]*(len(self.mp1_hidden_widths)-1)
        lst3 = [None]*(len(lst)+len(lst2))
        lst3[0::2] = lst
        lst3[1::2] = lst2
        self.Mc1_classifier = nn.Sequential(
                nn.Linear(self.fc_input_size, self.mp1_hidden_widths[0]), nn.ReLU(),
                *lst3,
                nn.Linear(self.mp1_hidden_widths[-1], len(self.first_classes)),
        )

        #self.second_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
        self.fc2_input_size = self.fc2_input_size + len(self.first_classes)
        lst = [nn.Linear(self.mp2_hidden_widths[i], self.mp2_hidden_widths[i+1]) for i in range(len(self.mp2_hidden_widths)-1) ]
        lst2 = [nn.ReLU()]*(len(self.mp2_hidden_widths)-1)
        lst3 = [None]*(len(lst)+len(lst2))
        lst3[0::2] = lst
        lst3[1::2] = lst2
        self.Mc2_classifier = nn.Sequential(
                nn.Linear(self.fc2_input_size, self.mp2_hidden_widths[0]), nn.ReLU(),
                *lst3,
                nn.Linear(self.mp2_hidden_widths[-1], len(self.second_classes)),
        )

        #self.third_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
        self.fc3_input_size = self.fc3_input_size + len(self.first_classes) 
        lst = [nn.Linear(self.mp3_hidden_widths[i], self.mp3_hidden_widths[i+1]) for i in range(len(self.mp3_hidden_widths)-1) ]
        lst2 = [nn.ReLU()]*(len(self.mp3_hidden_widths)-1)
        lst3 = [None]*(len(lst)+len(lst2))
        lst3[0::2] = lst
        lst3[1::2] = lst2
        self.Mc3_classifier = nn.Sequential(
                nn.Linear(self.fc3_input_size, self.mp3_hidden_widths[0]), nn.ReLU(),
                *lst3,
                nn.Linear(self.mp3_hidden_widths[-1], len(self.third_classes)),
        )


    def forward(self, X):
        # print([x for x in X])
        # print([X[x].shape for x in X if x != 'image'])

        if self.focus_on_largest_sunspot and not self.focus_on_group:
            # complete crop + largest sunspot mask

            one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['mask_LargestSpot']
            encoder_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)

        elif ((not self.focus_on_largest_sunspot) and self.focus_on_group):
            # (complete crop + group mask) OR (group crop + group mask)
            encoder_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:] * (X['group_mask']>0)], dim=1)
        
        else: 
            # (complete crop + complete mask)
            encoder_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:]], dim=1)

        encoder_output = self.encoder(encoder_input)
        # print([item.shape for item in encoder_output])

        ecoder_out_avg = self.globalAvgPooling(encoder_output[0])

        # print(ecoder_out_avg.shape)

        # to_concat = (ecoder_out_avg,) + tuple(X[input_type][:,:,None,None] for input_type in self.input_format[2:])
        to_concat = (ecoder_out_avg.squeeze(),) + tuple(X[input_type] for input_type in self.input_format[2:])
        classifiers_input = torch.cat(to_concat, axis=-1)
        # print(classifiers_input.shape)

        McI1 = self.Mc1_classifier(classifiers_input)

        fc2_in = torch.cat([classifiers_input, McI1], dim=1)
        McI2 = self.Mc2_classifier(fc2_in)
        
        fc3_in = torch.cat([classifiers_input, McI1], dim=1)
        McI3 = self.Mc3_classifier(fc3_in)


        return McI1, McI2, McI3