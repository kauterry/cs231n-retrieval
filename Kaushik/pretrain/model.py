from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch
from initialize_delf import init_delf, init_densenet_TL, init_delf_TL, init_resnet101gem, init_delf_pca

def initialize_model(model_name, num_classes, freeze_layers, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = init_densenet169()
        set_parameter_requires_grad(model_ft, freeze_layers)

        
    elif model_name == "densenet201":
        """ Densenet201
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze_layers)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    elif model_name == "delf":
        """ DELF using our pretrained Densenet169 features """
        model_ft = init_delf(num_classes)
        set_parameter_requires_grad(model_ft, 2)
        
    elif model_name == "delf_TL":
        """ DELF using our pretrained Densenet169 features, without FC layer """
        model_ft = init_delf_TL()
        set_parameter_requires_grad(model_ft, 2)
        
    elif model_name == "our_densenet_TL":
        """ Our pretrained Densenet169 without FC layer """
        model_ft = init_densenet_TL()
        set_parameter_requires_grad(model_ft, freeze_layers)
        
    elif model_name == "resnet101gem":
        model_ft = init_resnet101gem()
        set_parameter_requires_grad(model_ft, 0)
    
    elif model_name == "delf_pca":
        model_ft = init_delf_pca()
        set_parameter_requires_grad(model_ft, 1)
        
    else:
        print("Invalid model name, exiting...")
        exit()
    
#     model_ft = nn.Sequential(*list(model_ft.children()))
    
    return model_ft

def set_parameter_requires_grad(model, freeze_layers):        
    child_counter = 0
    for child in model.children():
        if child_counter < freeze_layers:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1
        
class Pool(nn.Module):
    def __init__(self, dim):
        super(Pool,self).__init__()
        self.dim = dim
        
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.dim)
        return out        

def init_densenet169():
    
    model_ft = models.densenet169(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2000)

    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(Pool(1664))
    features.append(list(model_ft.children())[-1])
    model_ft = nn.Sequential(*features) 
    
    return model_ft


        