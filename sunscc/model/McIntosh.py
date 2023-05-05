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

import matplotlib.pyplot as plt

from .NewResnet import ResNet, ResBlock, ResBottleneckBlock


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.
    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
        - Output: :math:`(ksize,)`
    """
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.
    Shape:
        - Output: :math:`(ksize_x, ksize_y)`
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def encoder_generator(architecture):
    if architecture['type'] == "mobilenet":
        mobilenets = {
            'v1': models.mobilenet, 
            'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large,
        }
        return mobilenets[architecture['mobilenet_version']](pretrained = architecture['pretrained'])
    elif  architecture['type'] == "resnet_torchvision":
        resnets = {
            '18': models.resnet18,
            '34': models.resnet34,
            '50': models.resnet50,
            '101': models.resnet101,
            '152': models.resnet152,
        }
        return resnets[architecture['resnet_version']](pretrained = architecture['pretrained'])
    elif  architecture['type'] == "vision_transformer":
        return models.vision_transformer.VisionTransformer(pretrained = architecture['pretrained'])
    elif  architecture['type'] == "new_resnet":
        resnets = {
            '18': (ResBlock, [2, 2, 2, 2]),
            '34': (ResBlock, [3, 4, 6, 3]),
            '50': (ResBottleneckBlock, [3, 4, 6, 3]),
            '101': (ResBottleneckBlock, [3, 4, 23, 3]),
        }
        in_channels = architecture['in_channels']
        blockType = resnets[str(architecture['resnet_version'])][0]
        layers = resnets[str(architecture['resnet_version'])][1]
        print("new_resnet", in_channels, blockType, layers)

        return ResNet( in_channels, blockType, layers)

class McIntoshClassifier_GenericOLD(nn.Module):
    def __init__(self, 
                # model_cfg,
                input_format,
                output_format,
                classes, 
                first_classes,
                second_classes,
                third_classes,
                num_channels_offset,
                cascade = False,
                double_encoder= False,
                architecture=None,
                focus_on_group=False,
                focus_on_largest_sunspot=False,
                parts_to_train=None,
                ):
        super().__init__()
        print("McIntoshClassifier_Generic")

        assert not (cascade and double_encoder)

        self.__dict__.update(locals())
        # mobilenets = {
        #     'v1': models.mobilenet, 
        #     'v2': models.mobilenet_v2,
        #     'v3_small': models.mobilenet_v3_small,
        #     'v3_large': models.mobilenet_v3_large,
        # }

        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        self.focus_on_group = focus_on_group
        self.focus_on_largest_sunspot = focus_on_largest_sunspot
        
        # self.mobilenet_version=self.architecture['mobilenet_version']
        # self.pretrained = self.architecture['pretrained']

        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        self.visual_input_format = self.input_format['visual']
        self.numeric_input_format = self.input_format['numeric']
        self.input_format = self.visual_input_format + self.numeric_input_format

        print(self.input_format, "///// .  /////////")

        # self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
        self.model = encoder_generator(self.architecture["encoder"])
        if self.architecture["encoder"]["type"] == "mobilenet":
            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()
            in_feats = self.model.classifier[1].in_features
        elif self.architecture["encoder"]["type"] == "resnet_torchvision":
            in_feats = self.model.fc.in_features
        elif self.architecture["encoder"]["type"] == "vision_transformer":
            in_feats = self.model.head.in_features
        elif self.architecture["encoder"]["type"] == "new_resnet":
            in_feats = self.model.fc.in_features
            self.model.fc = Identity()
        print("in_feats", in_feats)

        self.fc_input_size = in_feats + (len(self.numeric_input_format)) # concat inputs that are not 'image' and 'mask' 'mask_one_hot' 'mask_largestSpot'
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

        self.fc2_input_size = self.fc2_input_size + len(self.first_classes) if cascade else self.fc2_input_size
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
        
        self.fc3_input_size = self.fc3_input_size + len(self.first_classes) if cascade else self.fc3_input_size
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

        self.parts_to_train = parts_to_train
        print(parts_to_train)
        if parts_to_train is not None:
            if "encoder" not in parts_to_train:
                print("encoder -> requires_grad = False")
                for param in self.model.parameters():
                    param.requires_grad = False
            if "MLP1" not in parts_to_train:
                print("MLP1 -> requires_grad = False")
                for param in self.Mc1_classifier.parameters():
                    param.requires_grad = False
            if "MLP2" not in parts_to_train:
                print("MLP2 -> requires_grad = False")
                for param in self.Mc2_classifier.parameters():
                    param.requires_grad = False
            if "MLP3" not in parts_to_train:
                print("MLP3 -> requires_grad = False")
                for param in self.Mc3_classifier.parameters():
                    param.requires_grad = False

    def forward(self, X):
        # raise NotImplementedError
        # print(X.keys())

        img = X['image']
        excentricity = X['excentricity_map']

        grp_conf_map = X['group_confidence_map']

        # print('img', img.shape, 'excentricity', excentricity.shape, 'grp_conf_map', grp_conf_map.shape)

        encoder_input = torch.cat([img, excentricity, grp_conf_map], axis=1)
        # print('encoder_input', encoder_input.shape)
        
        # print('encoder_input', mobilenet_input.shape)
        encoder_output = self.model(encoder_input)
        # print('encoder_output', encoder_output.shape)

        to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.numeric_input_format)
        # print('to_concat', [x.shape for x in to_concat])
        classifiers_input = torch.cat(to_concat, axis=-1)
        # print('classifiers_input', classifiers_input.shape)

        McI1 = self.Mc1_classifier(classifiers_input)

        fc2_in = torch.cat([classifiers_input, McI1], dim=1)
        McI2 = self.Mc2_classifier(fc2_in)

        
        fc3_in = torch.cat([classifiers_input, McI1], dim=1)
        McI3 = self.Mc3_classifier(fc3_in)

        # print('McI1', McI1.shape, 'McI2', McI2.shape, 'McI3', McI3.shape)


        return McI1, McI2, McI3


    # def forwardOLD(self, X):
    #     # print([x for x in X])
    #     # print( f'Largest_sunspot:{self.focus_on_largest_sunspot} and Focus_group:{self.focus_on_group}')
    #     # print([X[x].shape for x in X if x != 'image'])
    #     if self.focus_on_largest_sunspot and not self.focus_on_group:
    #         # complete crop + largest sunspot mask
    #         # print("focus on largest sunspot and not group")

    #         # print("focus on largest sunspot")
    #         one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['mask_LargestSpot']
    #         mobilenet_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)


    #     elif self.focus_on_largest_sunspot and self.focus_on_group:
    #         # group crop + largest group sunspot mask
    #         # print("focus on largest sunspot in the group")

    #         one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['group_mask_LargestSpot']
    #         mobilenet_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)
            
    #     elif ((not self.focus_on_largest_sunspot) and self.focus_on_group):
    #         # (complete crop + group mask) OR (group crop + group mask)
    #         # print("focus on group and not largest sunspot")
    #         mobilenet_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:] * (X['group_mask']>0)], dim=1)
        
    #     else: 
    #         # (complete crop + complete mask)
    #         # print("focus on complete crop")
    #         mobilenet_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:]], dim=1)

    #     # print('encoder_input', mobilenet_input.shape)
    #     encoder_output = self.model(mobilenet_input)

    #     to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.numeric_input_format)
    #     classifiers_input = torch.cat(to_concat, axis=-1)

    #     McI1 = self.Mc1_classifier(classifiers_input)

    #     fc2_in = torch.cat([classifiers_input, McI1], dim=1)
    #     McI2 = self.Mc2_classifier(fc2_in)
        
    #     fc3_in = torch.cat([classifiers_input, McI1], dim=1)
    #     McI3 = self.Mc3_classifier(fc3_in)


    #     return McI1, McI2, McI3


class McIntoshClassifier_Generic(nn.Module):
    def __init__(self, 
                # model_cfg,
                input_format,
                output_format,
                classes, 
                first_classes,
                second_classes,
                third_classes,
                num_channels_offset,
                cascade = False,
                double_encoder= False,
                architecture=None,
                focus_on_group=False,
                focus_on_largest_sunspot=False,
                parts_to_train=None,
                ):
        super().__init__()
        print("McIntoshClassifier_Generic")

        assert not (cascade and double_encoder)

        self.__dict__.update(locals())
        # mobilenets = {
        #     'v1': models.mobilenet, 
        #     'v2': models.mobilenet_v2,
        #     'v3_small': models.mobilenet_v3_small,
        #     'v3_large': models.mobilenet_v3_large,
        # }
        self.cascade = cascade

        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        self.focus_on_group = focus_on_group
        self.focus_on_largest_sunspot = focus_on_largest_sunspot
        
        # self.mobilenet_version=self.architecture['mobilenet_version']
        # self.pretrained = self.architecture['pretrained']

        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        self.visual_input_format = self.input_format['visual']
        self.numeric_input_format = self.input_format['numeric']
        self.input_format = self.visual_input_format + self.numeric_input_format

        print(self.input_format, "///// .  /////////")

        # self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
        self.model = encoder_generator(self.architecture["encoder"])
        if self.architecture["encoder"]["type"] == "mobilenet":
            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()
            in_feats = self.model.classifier[1].in_features
        elif self.architecture["encoder"]["type"] == "resnet_torchvision":
            in_feats = self.model.fc.in_features
        elif self.architecture["encoder"]["type"] == "vision_transformer":
            in_feats = self.model.head.in_features
        elif self.architecture["encoder"]["type"] == "new_resnet":
            in_feats = self.model.fc.in_features
            self.model.fc = Identity()
        print("in_feats", in_feats)

        self.fc_input_size = in_feats + (len(self.numeric_input_format)) # concat inputs that are not 'image' and 'mask' 'mask_one_hot' 'mask_largestSpot'
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
        print("self.fc1_input_size", self.fc_input_size)

        self.fc2_input_size = self.fc2_input_size + len(self.first_classes) if cascade else self.fc2_input_size
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
        print("self.fc2_input_size", self.fc2_input_size)
        
        self.fc3_input_size = self.fc3_input_size + len(self.first_classes) if cascade else self.fc3_input_size
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
        print("self.fc3_input_size", self.fc3_input_size)

        self.parts_to_train = parts_to_train
        print(parts_to_train)
        if parts_to_train is not None:
            if "encoder" not in parts_to_train:
                print("encoder -> requires_grad = False")
                for param in self.model.parameters():
                    param.requires_grad = False
            if "MLP1" not in parts_to_train:
                print("MLP1 -> requires_grad = False")
                for param in self.Mc1_classifier.parameters():
                    param.requires_grad = False
            if "MLP2" not in parts_to_train:
                print("MLP2 -> requires_grad = False")
                for param in self.Mc2_classifier.parameters():
                    param.requires_grad = False
            if "MLP3" not in parts_to_train:
                print("MLP3 -> requires_grad = False")
                for param in self.Mc3_classifier.parameters():
                    param.requires_grad = False

    def forward(self, X):
        # raise NotImplementedError
        # print(X.keys())

        img = None
        if 'image' in self.visual_input_format:
            img = X['image']

        excentricity = None
        if 'excentricity_map' in self.visual_input_format:
            excentricity = X['excentricity_map']

        grp_conf_map = None
        if 'group_confidence_map' in self.visual_input_format:
            grp_conf_map = X['group_confidence_map']

        lst = [img, excentricity, grp_conf_map]
        encoder_input = torch.cat([x for x in lst if x is not None], axis=1)
        # print('img', img.shape, 'excentricity', excentricity.shape, 'grp_conf_map', grp_conf_map.shape)

        # encoder_input = torch.cat([img, excentricity, grp_conf_map], axis=1)
        # # print('encoder_input', encoder_input.shape)
        
        # print('encoder_input', mobilenet_input.shape)
        encoder_output = self.model(encoder_input)
        # print('encoder_output', encoder_output.shape)

        to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.numeric_input_format)
        # print('to_concat', [x.shape for x in to_concat])
        classifiers_input = torch.cat(to_concat, axis=-1)
        # print('classifiers_input', classifiers_input.shape)

        McI1 = self.Mc1_classifier(classifiers_input)
        # McI1 = self.Mc1_classifier(classifiers_input).clone().detach()
        McI1_other = McI1.clone().detach()

        if self.cascade:
            # print('cascade')
            fc2_in = torch.cat([classifiers_input, McI1_other], dim=1)
            fc3_in = torch.cat([classifiers_input, McI1_other], dim=1)
            # fc2_in = torch.cat([classifiers_input, McI1], dim=1)
            # fc3_in = torch.cat([classifiers_input, McI1], dim=1)
        else:
            fc2_in = classifiers_input
            fc3_in = classifiers_input

        McI2 = self.Mc2_classifier(fc2_in)

    
        McI3 = self.Mc3_classifier(fc3_in)

        # print('McI1', McI1.shape, 'McI2', McI2.shape, 'McI3', McI3.shape)


        return McI1, McI2, McI3




class McIntoshClassifier_MobilenetEncoder(nn.Module):
    def __init__(self, 
                # model_cfg,
                input_format,
                output_format,
                classes, 
                first_classes,
                second_classes,
                third_classes,
                num_channels_offset,
                cascade = False,
                double_encoder= False,
                architecture=None,
                focus_on_group=False,
                focus_on_largest_sunspot=False,
                ):
        super().__init__()

        assert not (cascade and double_encoder)

        self.__dict__.update(locals())
        mobilenets = {
            'v1': models.mobilenet, 
            'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large,
        }
        self.cascade = cascade
        self.double_encoder = double_encoder

        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        self.focus_on_group = focus_on_group
        self.focus_on_largest_sunspot = focus_on_largest_sunspot
        
        self.mobilenet_version=self.architecture['mobilenet_version']
        self.pretrained = self.architecture['pretrained']

        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        self.visual_input_format = self.input_format['visual']
        self.numeric_input_format = self.input_format['numeric']
        self.input_format = self.visual_input_format + self.numeric_input_format

        print(self.input_format, "///// .  /////////")

        self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
        
        in_feats = self.model.classifier[1].in_features
        self.fc_input_size = in_feats + (len(self.numeric_input_format)) # concat inputs that are not 'image' and 'mask' 'mask_one_hot' 'mask_largestSpot'
        self.fc2_input_size = self.fc_input_size
        self.fc3_input_size = self.fc_input_size

        # The classifier of the model should be removed -> replace with identity
        # This way, the mobilenet model is used as a simple encoder.
        self.model.classifier = Identity()

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

        self.fc2_input_size = self.fc2_input_size + len(self.first_classes) if cascade else self.fc2_input_size
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
        
        self.fc3_input_size = self.fc3_input_size + len(self.first_classes) if cascade else self.fc3_input_size
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
        # print( f'Largest_sunspot:{self.focus_on_largest_sunspot} and Focus_group:{self.focus_on_group}')
        # print([X[x].shape for x in X if x != 'image'])
        if self.focus_on_largest_sunspot and not self.focus_on_group:
            # complete crop + largest sunspot mask
            # print("focus on largest sunspot and not group")

            # print("focus on largest sunspot")
            one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['mask_LargestSpot']
            mobilenet_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)


        elif self.focus_on_largest_sunspot and self.focus_on_group:
            # group crop + largest group sunspot mask
            # print("focus on largest sunspot in the group")

            one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['group_mask_LargestSpot']
            mobilenet_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)
            
        elif ((not self.focus_on_largest_sunspot) and self.focus_on_group):
            # (complete crop + group mask) OR (group crop + group mask)
            # print("focus on group and not largest sunspot")
            mobilenet_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:] * (X['group_mask']>0)], dim=1)
        
        else: 
            # (complete crop + complete mask)
            # print("focus on complete crop")
            mobilenet_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:]], dim=1)

        # print('encoder_input', mobilenet_input.shape)
        encoder_output = self.model(mobilenet_input)

        to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.numeric_input_format)
        classifiers_input = torch.cat(to_concat, axis=-1)

        McI1 = self.Mc1_classifier(classifiers_input)

        fc2_in = torch.cat([classifiers_input, McI1], dim=1)
        McI2 = self.Mc2_classifier(fc2_in)
        
        fc3_in = torch.cat([classifiers_input, McI1], dim=1)
        McI3 = self.Mc3_classifier(fc3_in)


        return McI1, McI2, McI3



class McIntoshClassifier(nn.Module):
    def __init__(self, 
                # model_cfg,
                input_format,
                output_format,
                classes, 
                first_classes,
                second_classes,
                third_classes,
                num_channels_offset,
                cascade = False,
                double_encoder= False,
                architecture=None,
                focus_on_largest_sunspot=False,
                ):
        super().__init__()

        assert not (cascade and double_encoder)

        self.__dict__.update(locals())
        mobilenets = {
            'v1': models.mobilenet, 
            'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large,
        }
        self.cascade = cascade
        self.double_encoder = double_encoder

        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        self.focus_on_largest_sunspot = focus_on_largest_sunspot
        
        self.mobilenet_version=self.architecture['mobilenet_version']
        self.pretrained = self.architecture['pretrained']

        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        self.visual_input_format = self.input_format['visual']
        self.numeric_input_format = self.input_format['numeric']
        self.input_format = self.visual_input_format + self.numeric_input_format

        print(self.input_format, "///// .  /////////")

        # self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
        if double_encoder:

            self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)

            in_feats = self.model.classifier[1].in_features

            self.fc_input_size = in_feats + (len(self.input_format)-2) # concat inputs that are not 'image' and 'mask'
            self.fc2_input_size = self.model.features[6].out_channels
            self.third_fc_input_size = self.fc_input_size

            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()

            # get  last dense layer of encoder1
            # return_nodes = ['features.6','flatten']
            self.return_nodes = ['features.'+str(i) for i in range(18)]+['flatten']

            self.model = create_feature_extractor(self.model, return_nodes=self.return_nodes)

            # print(self.model)
            
            self.hidden_width = 100
            # Classifiers
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
            # self.Mc1_classifier = nn.Sequential(
            #         nn.Linear(self.fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.first_classes)),
            # )
            
            print("self.fc2_input_size", self.fc2_input_size)
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
            # self.Mc2_classifier = nn.Sequential(
            #         nn.Linear(self.fc2_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.second_classes)),
            # )
            
            lst = [nn.Linear(self.mp3_hidden_widths[i], self.mp3_hidden_widths[i+1]) for i in range(len(self.mp3_hidden_widths)-1) ]
            lst2 = [nn.ReLU()]*(len(self.mp3_hidden_widths)-1)
            lst3 = [None]*(len(lst)+len(lst2))
            lst3[0::2] = lst
            lst3[1::2] = lst2
            self.Mc3_classifier = nn.Sequential(
                    nn.Linear(self.third_fc_input_size, self.mp3_hidden_widths[0]), nn.ReLU(),
                    *lst3,
                    nn.Linear(self.mp3_hidden_widths[-1], len(self.third_classes)),
            )
            # self.Mc3_classifier = nn.Sequential(
            #         nn.Linear(self.third_fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.third_classes)),
            # )

        else:        
            
            self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
            # first_layer = self.model.features[0][0] 
            # # print(first_layer)
            # # print(first_layer.in_channels, first_layer.out_channels, first_layer.kernel_size, first_layer.stride, 
            # #     first_layer.padding, first_layer.bias)
            # new_layer = nn.Conv2d(in_channels=self.num_channels, 
            #                     out_channels=first_layer.out_channels, 
            #                     kernel_size=first_layer.kernel_size,
            #                     stride=first_layer.stride, 
            #                     padding=first_layer.padding,
            #                     bias=first_layer.bias)
            # self.model.features[0][0]= new_layer
            # # print(self.model.features[0][0])
            
            in_feats = self.model.classifier[1].in_features
            # self.fc_input_size = in_feats + (len(self.input_format)-2) # concat inputs that are not 'image' and 'mask'
            # self.fc_input_size = in_feats + (len(self.input_format)-3) # concat inputs that are not 'image' and 'mask' 'mask_one_hot'
            # self.fc_input_size = in_feats + (len(self.input_format)-4) # concat inputs that are not 'image' and 'mask' 'mask_one_hot' 'mask_largestSpot'
            self.fc_input_size = in_feats + (len(self.numeric_input_format)) # concat inputs that are not 'image' and 'mask' 'mask_one_hot' 'mask_largestSpot'
            self.fc2_input_size = self.fc_input_size
            self.fc3_input_size = self.fc_input_size

            
            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()

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
            # self.Mc1_classifier = nn.Sequential(
            #         nn.Linear(self.fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.first_classes)),
            # )

            #self.second_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
            self.fc2_input_size = self.fc2_input_size + len(self.first_classes) if cascade else self.fc2_input_size
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
            # self.Mc2_classifier = nn.Sequential(
            #         nn.Linear(self.fc2_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.second_classes)),
            # )

            #self.third_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
            self.fc3_input_size = self.fc3_input_size + len(self.first_classes) if cascade else self.fc3_input_size
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
            # self.Mc3_classifier = nn.Sequential(
            #         nn.Linear(self.fc3_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.third_classes)),
            # )

    def reduce_mask(self, mask, kside, sigma, resize_side):
        kernel = get_gaussian_kernel2d(ksize=(kside,kside),sigma=(sigma,sigma)).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(device=mask.device)

        # print(kernel.device, mask.device)

        padder = T.Pad(padding=kside//2)
        msk_blurred = F.conv2d(padder(mask.float()), kernel)

        resizer = T.Resize((resize_side,resize_side))
        res_lowpass = resizer(msk_blurred.squeeze(0))
        
        norm_res_lowpass = res_lowpass / res_lowpass.max()
        
        # print(res_lowpass.min(), res_lowpass.max(), '->', norm_res_lowpass.min(), norm_res_lowpass.max())
        # return res_lowpass
        return norm_res_lowpass

    def compute_vector(self, feats, mask):
        # print(feats.shape , mask.shape)
        sum_Wij = mask.sum()
        assert sum_Wij > 0
        
        prod = feats * mask
        sum_CijWij = torch.sum(prod, axis=(2,3))
        # print(sum_CijWij.shape)
    #     sum_CijWij = np.sum(feats * reduced_mask[None,:,:,:], axis=(2,3))
        
        out_arr = sum_CijWij / sum_Wij
        
        return out_arr, prod



    def forward(self, X):
        # print([x for x in X])
        # print([X[x].shape for x in X if x != 'image'])

        
        if  (not self.cascade) and (not self.double_encoder):
            mobilenet_input = X['image']
            # mobilenet_input = torch.cat([X['image'], X['mask']], dim=1)
            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)


            to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[2:])
            classifiers_input = torch.cat(to_concat, axis=-1)

            McI1 = self.Mc1_classifier(classifiers_input)
            McI2 = self.Mc2_classifier(classifiers_input)
            McI3 = self.Mc3_classifier(classifiers_input)

        elif (self.cascade) and (not self.double_encoder):
            # mobilenet_input = X['image']

            if self.focus_on_largest_sunspot:
                # print("focus on largest sunspot")
                one_hot_largestSpot = X['mask_one_hot'][:,1:,:,:] * X['mask_LargestSpot']
                mobilenet_input = torch.cat([X['image'], one_hot_largestSpot], dim=1)
            else:
                # print('whole segmentation')
                mobilenet_input = torch.cat([X['image'], X['mask_one_hot'][:,1:,:,:]], dim=1)

            
            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)

            # to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[2:])
            # to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[3:])
            # to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[4:])

            to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.numeric_input_format)
            classifiers_input = torch.cat(to_concat, axis=-1)

            McI1 = self.Mc1_classifier(classifiers_input)

            fc2_in = torch.cat([classifiers_input, McI1], dim=1)
            McI2 = self.Mc2_classifier(fc2_in)
            
            fc3_in = torch.cat([classifiers_input, McI1], dim=1)
            McI3 = self.Mc3_classifier(fc3_in)

        elif  (not self.cascade) and ( self.double_encoder):

            # mobilenet_input = X['image']
            mobilenet_input = X['image']
            mask_input = X['mask']

            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)

            # for out in encoder_output:
            #     print(out, encoder_output[out].shape)


            # Use last output of the encoder as 1st and 3rd classifiers input
            firstAndThird_input = encoder_output[list(encoder_output.keys())[-1]]
            # print(firstAndThird_input.shape)
            # concatenate with inputs that are not 'mask' and 'image'
            to_concat = (firstAndThird_input,) + tuple(X[input_type] for input_type in self.input_format[2:])
            firstAndThird_input = torch.cat(to_concat, axis=-1)
            # print(firstAndThird_input.shape)
            
            # Use last dense layer output of encoder + reduced blurred mask -> encode it -> as 2nd classifier input
            second_input = encoder_output[list(encoder_output.keys())[list(encoder_output.keys()).index('features.6')]]
            
            kside = mask_input.shape[-1] - 1 if mask_input.shape[-1]%2==0  else mask_input.shape[-1] 
            sigma = 10
            reduced_masks = self.reduce_mask(mask_input,kside,sigma, second_input.shape[-1])
            McI2_input_vector, _ = self.compute_vector(second_input, reduced_masks)

            # raise

            McI1 = self.Mc1_classifier(firstAndThird_input)
            McI2 = self.Mc2_classifier(McI2_input_vector)
            McI3 = self.Mc3_classifier(firstAndThird_input)

        return McI1, McI2, McI3

class McIntoshClassifierOLD(nn.Module):
    def __init__(self, 
                model_cfg,
                input_format,
                output_format,
                classes, 
                first_classes,
                second_classes,
                third_classes,
                num_channels_offset,
                cascade = False,
                double_encoder= False,
                architecture=None,
                focus_on_largest_sunspot=False,
                ):
        super().__init__()

        assert not (cascade and double_encoder)

        self.__dict__.update(locals())
        mobilenets = {
            'v1': models.mobilenet, 
            'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large,
        }
        self.cascade = cascade
        self.double_encoder = double_encoder

        self.classes = classes

        self.first_classes = first_classes
        self.second_classes = second_classes
        self.third_classes = third_classes

        self.architecture = architecture
        
        self.mobilenet_version=self.architecture['mobilenet_version']
        self.pretrained = self.architecture['pretrained']

        self.mp1_hidden_widths = self.architecture["MLP1"]
        self.mp2_hidden_widths = self.architecture["MLP2"]
        self.mp3_hidden_widths = self.architecture["MLP3"]

        
        print(self.input_format, "///// .  /////////")

        # self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
        if double_encoder:

            self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)

            in_feats = self.model.classifier[1].in_features

            self.fc_input_size = in_feats + (len(self.input_format)-2) # concat inputs that are not 'image' and 'mask'
            self.fc2_input_size = self.model.features[6].out_channels
            self.third_fc_input_size = self.fc_input_size

            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()

            # get  last dense layer of encoder1
            # return_nodes = ['features.6','flatten']
            self.return_nodes = ['features.'+str(i) for i in range(18)]+['flatten']

            self.model = create_feature_extractor(self.model, return_nodes=self.return_nodes)

            # print(self.model)
            
            self.hidden_width = 100
            # Classifiers
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
            # self.Mc1_classifier = nn.Sequential(
            #         nn.Linear(self.fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.first_classes)),
            # )
            
            print("self.fc2_input_size", self.fc2_input_size)
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
            # self.Mc2_classifier = nn.Sequential(
            #         nn.Linear(self.fc2_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.second_classes)),
            # )
            
            lst = [nn.Linear(self.mp3_hidden_widths[i], self.mp3_hidden_widths[i+1]) for i in range(len(self.mp3_hidden_widths)-1) ]
            lst2 = [nn.ReLU()]*(len(self.mp3_hidden_widths)-1)
            lst3 = [None]*(len(lst)+len(lst2))
            lst3[0::2] = lst
            lst3[1::2] = lst2
            self.Mc3_classifier = nn.Sequential(
                    nn.Linear(self.third_fc_input_size, self.mp3_hidden_widths[0]), nn.ReLU(),
                    *lst3,
                    nn.Linear(self.mp3_hidden_widths[-1], len(self.third_classes)),
            )
            # self.Mc3_classifier = nn.Sequential(
            #         nn.Linear(self.third_fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.third_classes)),
            # )

        else:        
            
            self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
            # first_layer = self.model.features[0][0] 
            # # print(first_layer)
            # # print(first_layer.in_channels, first_layer.out_channels, first_layer.kernel_size, first_layer.stride, 
            # #     first_layer.padding, first_layer.bias)
            # new_layer = nn.Conv2d(in_channels=self.num_channels, 
            #                     out_channels=first_layer.out_channels, 
            #                     kernel_size=first_layer.kernel_size,
            #                     stride=first_layer.stride, 
            #                     padding=first_layer.padding,
            #                     bias=first_layer.bias)
            # self.model.features[0][0]= new_layer
            # # print(self.model.features[0][0])
            
            in_feats = self.model.classifier[1].in_features
            self.fc_input_size = in_feats + (len(self.input_format)-2) # concat inputs that are not 'image' and 'mask'
            
            self.fc2_input_size = self.fc_input_size
            self.fc3_input_size = self.fc_input_size

            
            # The classifier of the model should be removed -> replace with identity
            # This way, the mobilenet model is used as a simple encoder.
            self.model.classifier = Identity()

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
            # self.Mc1_classifier = nn.Sequential(
            #         nn.Linear(self.fc_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.first_classes)),
            # )

            #self.second_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
            self.fc2_input_size = self.fc2_input_size + len(self.first_classes) if cascade else self.fc2_input_size
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
            # self.Mc2_classifier = nn.Sequential(
            #         nn.Linear(self.fc2_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.second_classes)),
            # )

            #self.third_fc_input_size = self.fc_input_size + self.Mc1_classifier.out_features if cascade else self.fc_input_size
            self.fc3_input_size = self.fc3_input_size + len(self.first_classes) if cascade else self.fc3_input_size
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
            # self.Mc3_classifier = nn.Sequential(
            #         nn.Linear(self.fc3_input_size, self.hidden_width), nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, self.hidden_width),  nn.ReLU(),
            #         nn.Linear(self.hidden_width, len(self.third_classes)),
            # )

    def reduce_mask(self, mask, kside, sigma, resize_side):
        kernel = get_gaussian_kernel2d(ksize=(kside,kside),sigma=(sigma,sigma)).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(device=mask.device)

        # print(kernel.device, mask.device)

        padder = T.Pad(padding=kside//2)
        msk_blurred = F.conv2d(padder(mask.float()), kernel)

        resizer = T.Resize((resize_side,resize_side))
        res_lowpass = resizer(msk_blurred.squeeze(0))
        
        norm_res_lowpass = res_lowpass / res_lowpass.max()
        
        # print(res_lowpass.min(), res_lowpass.max(), '->', norm_res_lowpass.min(), norm_res_lowpass.max())
        # return res_lowpass
        return norm_res_lowpass

    def compute_vector(self, feats, mask):
        # print(feats.shape , mask.shape)
        sum_Wij = mask.sum()
        assert sum_Wij > 0
        
        prod = feats * mask
        sum_CijWij = torch.sum(prod, axis=(2,3))
        # print(sum_CijWij.shape)
    #     sum_CijWij = np.sum(feats * reduced_mask[None,:,:,:], axis=(2,3))
        
        out_arr = sum_CijWij / sum_Wij
        
        return out_arr, prod



    def forward(self, X):
        # print([x for x in X])
        # print([X[x].shape for x in X if x != 'image'])

        
        if  (not self.cascade) and (not self.double_encoder):
            mobilenet_input = X['image']
            # mobilenet_input = torch.cat([X['image'], X['mask']], dim=1)
            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)


            to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[2:])
            classifiers_input = torch.cat(to_concat, axis=-1)

            McI1 = self.Mc1_classifier(classifiers_input)
            McI2 = self.Mc2_classifier(classifiers_input)
            McI3 = self.Mc3_classifier(classifiers_input)

        elif (self.cascade) and (not self.double_encoder):
            mobilenet_input = X['image']

            
            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)

            to_concat = (encoder_output,) + tuple(X[input_type] for input_type in self.input_format[2:])
            
            classifiers_input = torch.cat(to_concat, axis=-1)

            McI1 = self.Mc1_classifier(classifiers_input)

            fc2_in = torch.cat([classifiers_input, McI1], dim=1)
            McI2 = self.Mc2_classifier(fc2_in)
            
            fc3_in = torch.cat([classifiers_input, McI1], dim=1)
            McI3 = self.Mc3_classifier(fc3_in)

        elif  (not self.cascade) and ( self.double_encoder):

            # mobilenet_input = X['image']
            mobilenet_input = X['image']
            mask_input = X['mask']

            # print('encoder_input', mobilenet_input.shape)
            encoder_output = self.model(mobilenet_input)

            # for out in encoder_output:
            #     print(out, encoder_output[out].shape)


            # Use last output of the encoder as 1st and 3rd classifiers input
            firstAndThird_input = encoder_output[list(encoder_output.keys())[-1]]
            # print(firstAndThird_input.shape)
            # concatenate with inputs that are not 'mask' and 'image'
            to_concat = (firstAndThird_input,) + tuple(X[input_type] for input_type in self.input_format[2:])
            firstAndThird_input = torch.cat(to_concat, axis=-1)
            # print(firstAndThird_input.shape)
            
            # Use last dense layer output of encoder + reduced blurred mask -> encode it -> as 2nd classifier input
            second_input = encoder_output[list(encoder_output.keys())[list(encoder_output.keys()).index('features.6')]]
            
            kside = mask_input.shape[-1] - 1 if mask_input.shape[-1]%2==0  else mask_input.shape[-1] 
            sigma = 10
            reduced_masks = self.reduce_mask(mask_input,kside,sigma, second_input.shape[-1])
            McI2_input_vector, _ = self.compute_vector(second_input, reduced_masks)

            # raise

            McI1 = self.Mc1_classifier(firstAndThird_input)
            McI2 = self.Mc2_classifier(McI2_input_vector)
            McI3 = self.Mc3_classifier(firstAndThird_input)

        return McI1, McI2, McI3
        