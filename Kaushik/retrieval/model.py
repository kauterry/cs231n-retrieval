from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from delf import delf_densenet169, init_resnet101gem, init_delf_pca

def initialize_model(model_name, num_classes = 2000, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.features.parameters():
            param.requires_grad = False
            
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.features.parameters():
            param.requires_grad = False

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.features.parameters():
            param.requires_grad = False
        features = list(model_ft.children())[:-1]
        features.append(nn.ReLU(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        model_ft = nn.Sequential(*features)
        
    elif model_name == "resnext50":
        """ Resnext50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.features.parameters():
            param.requires_grad = False
            
    elif model_name == "resnext101":
        """ Resnext101
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.features.parameters():
            param.requires_grad = False

    elif model_name == "delf":
        """ DELF using pretrained Densenet169 features """
        model_ft = delf_densenet169()  
#         num_ftrs = model_ft.classifier.in_features
#         model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        for param in model_ft.parameters():
            param.requires_grad = False     
            
    elif model_name == "resnet101gem":
        model_ft = init_resnet101gem()
        for param in model_ft.parameters():
            param.requires_grad = False  
            
    elif model_name == "delf_pca":
        model_ft = init_delf_pca()
        for param in model_ft.parameters():
            param.requires_grad = False  
    elif model_name == "densenet_class":
        model_ft = init_densenet169()
        for param in model_ft.parameters():
            param.requires_grad = False
    else:
        print("Invalid model name, exiting...")
        exit()
    
#     model_ft = nn.Sequential(*list(model_ft.children())[:-1])
    
    return model_ft

    
def load_model(model_name, model_path):
    
    model = initialize_model(model_name, use_pretrained=True)
    
    # Send the model to GPU
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        print("Let's use CPU!")

    model_dict = model.state_dict()
#     print (model_dict.keys())
    load_dict = torch.load(model_path)
#     print ("-"*100)
#     print (load_dict.keys())
    
    model_dict = {k: v for k, v in load_dict.items() if k in model_dict}
    model.load_state_dict(model_dict)
    
    return model
    
 


class Pool(nn.Module):
    def __init__(self, dim):
        super(Pool,self).__init__()
        self.dim = dim

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.dim)
        return out        


def init_densenet169():
    
    model_ft = models.densenet169(pretrained=True)
#     num_ftrs = model_ft.classifier.in_features
#     model_ft.classifier = nn.Linear(num_ftrs, 2000)

    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(Pool(1664))
#     features.append(list(model_ft.children())[-1])
    model_ft = nn.Sequential(*features) 
    
    return model_ft
