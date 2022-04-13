from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets.folder import make_dataset

from torchsummary import summary
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from ptflops import get_model_complexity_info


def alexnet(num_classes):

    model_ft = models.alexnet(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def vgg11(num_classes):

    model_ft = models.vgg11(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def vgg13(num_classes):

    model_ft = models.vgg13(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def vgg16(num_classes):

    model_ft = models.vgg16(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def vgg19(num_classes):

    model_ft = models.vgg19(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet18(num_classes):

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet34(num_classes):

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet50(num_classes):

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet101(num_classes):

    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnet152(num_classes):

    model_ft = models.resnet152(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def squeezenet1_0(num_classes):

    model_ft = models.squeezenet1_0(pretrained=True)
    num_ftrs = model_ft.classifier[1].in_channels
    model_ft.classifier[1]= nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model_ft

def squeezenet1_1(num_classes):

    model_ft = models.squeezenet1_1(pretrained=True)
    num_ftrs = model_ft.classifier[1].in_channels
    model_ft.classifier[1]= nn.Conv2d(num_ftrs, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model_ft

def densenet121(num_classes):

    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def densenet169(num_classes):

    model_ft = models.densenet169(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def densenet161(num_classes):

    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def densenet201(num_classes):

    model_ft = models.densenet201(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def inception_v3(num_classes):

    model_ft = models.inception_v3(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def googlenet(num_classes):

    model_ft = models.googlenet(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def shufflenet_v2_x0_5(num_classes):

    model_ft = models.shufflenet_v2_x0_5(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def shufflenet_v2_x1_0(num_classes):

    model_ft = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def shufflenet_v2_x1_5(num_classes):

    model_ft = models.shufflenet_v2_x1_5(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def shufflenet_v2_x2_0(num_classes):

    model_ft = models.shufflenet_v2_x2_0(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def mobilenet_v2(num_classes):

    model_ft = models.mobilenet_v2(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def mobilenet_v3_large(num_classes):

    model_ft = models.mobilenet_v3_large(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def mobilenet_v3_small(num_classes):

    model_ft = models.mobilenet_v3_small(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def resnext50_32x4d(num_classes):

    model_ft = models.resnext50_32x4d(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft

def resnext101_32x8d(num_classes):

    model_ft = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def wide_resnet50_2(num_classes):

    model_ft = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def wide_resnet101_2(num_classes):

    model_ft = models.wide_resnet101_2(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def mnasnet0_5(num_classes):

    model_ft = models.mnasnet0_5(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def mnasnet0_75(num_classes):

    model_ft = models.mnasnet0_75(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def mnasnet1_0(num_classes):

    model_ft = models.mnasnet1_0(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def mnasnet1_3(num_classes):

    model_ft = models.mnasnet1_3(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b0(num_classes):

    model_ft = models.efficientnet_b0(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b1(num_classes):

    model_ft = models.efficientnet_b1(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b2(num_classes):

    model_ft = models.efficientnet_b2(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b3(num_classes):

    model_ft = models.efficientnet_b3(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b4(num_classes):

    model_ft = models.efficientnet_b4(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b5(num_classes):

    model_ft = models.efficientnet_b5(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b6(num_classes):

    model_ft = models.efficientnet_b6(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def efficientnet_b7(num_classes):

    model_ft = models.efficientnet_b7(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def convnext_tiny(num_classes):

    model_ft = models.convnext_tiny(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft  

def convnext_small(num_classes):

    model_ft = models.convnext_small(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def convnext_base(num_classes):

    model_ft = models.convnext_base(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def convnext_large(num_classes):

    model_ft = models.convnext_large(pretrained=True)
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1]= nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft 

def resnest50(num_classes):

    model_ft = resnest50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnest101(num_classes):

    model_ft = resnest101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnest200(num_classes):

    model_ft = resnest200(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

def resnest269(num_classes):

    model_ft = resnest269(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft

if __name__ == '__main__':
    # model = alexnet(5)
    # model.cuda()
    # summary(model,input_size=(3,256,256))
    # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    model_ft = models.regnet_y_400mf(pretrained=True)
    print(model_ft) 




