import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, utils, models
from torch.nn.parameter import Parameter


def load_densenet(model_path, num_classes):
    
    model_ft = models.densenet169(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(nn.AdaptiveAvgPool2d((1, 1)))
    features.append(list(model_ft.children())[-1])

    model = nn.Sequential(*features)
    model = nn.DataParallel(model)
    model_dict = model.state_dict()
    load_dict = torch.load(model_path)
    model_dict = {k: v for k, v in load_dict.items() if k in model_dict}
    model.load_state_dict(model_dict)

    return model.module #.module to remove model from DataParallel



class AttentionPool(torch.nn.Module):
    def __init__(self, d_in):
        super(AttentionPool, self).__init__()
        self.conv1 = nn.Conv2d(d_in, 512, 1)
        self.conv2 = nn.Conv2d(512, 1, 1)
        with torch.no_grad():
            self.conv2.bias += torch.ones(1)
        self.activ = nn.Softplus()
        
    def forward(self, x):
        scores = self.conv2(self.conv1(x).clamp(min=0))
        weights = self.activ(scores)
        wtd_feat = x * weights
        out_feat = torch.mean(wtd_feat, dim=(2,3))
        return out_feat
        
class debugpool(torch.nn.Module):
    def __init(self):
        super(debugpool, self).__init__()
        
    def forward(self, x):
        output = nn.AvgPool2d(7)(x)
        output = torch.squeeze(output)
        return output
        
def init_delf(num_classes):
    model_path = '/home/kylecshan/saved/06_01_2019/classification/densenet169/model1_epoch9.pth'
    output_dim = 1664
    
    model = load_densenet(model_path, num_classes)
    delf_model = nn.Sequential(*list(model.children())[:-2],
                               #debugpool(),
                               AttentionPool(output_dim),
                               list(model.children())[-1])
    return delf_model
    
    
def init_densenet_TL():
    model_path = '/home/kylecshan/saved/05_16_2019/model815.pth'
    output_dim = 1664
    
    model = load_densenet(model_path, 2000)
    model = nn.Sequential(*list(model.children())[:-1])
    return model


def init_delf_TL():
    model_path = '/home/kylecshan/saved/06_01_2019/classification/densenet169/model1_epoch9.pth'
    output_dim = 1664
    
    model = load_densenet(model_path, 2000)
    delf_model = nn.Sequential(*list(model.children())[:-2],
                               AttentionPool(output_dim))
    return delf_model

def init_delf_pca():
    model_path = '/home/kylecshan/saved/06_01_2019/triplet/delf/model1_epoch7.pth'
    load_dict = torch.load(model_path)

    model_ft = models.densenet169(pretrained=True)
    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(AttentionPool(1664))

    model = nn.Sequential(*features)

    model = nn.DataParallel(model)

    model_dict = model.state_dict()

    model_dict = {k: v for k, v in load_dict.items() if k in model_dict}
    model.load_state_dict(model_dict)

    delf_pca_model = nn.Sequential(*list(model.children()),
                                       nn.Linear(1664, 768))
    return delf_pca_model


def gem(x, p=3, eps=1e-6):
     
    pool = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    l2n = pool / (torch.norm(pool, p=2, dim=1, keepdim=True) + eps).expand_as(pool)
    flatten = l2n.squeeze(-1).squeeze(-1)

    return flatten
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

class Whiten(nn.Module):
    def __init__(self, dim = 2048, eps=1e-6):
        super(Whiten,self).__init__()
        self.whiten = nn.Linear(dim, dim, bias = True)
        self.eps = eps
        
    def forward(self, x):
        out = self.whiten(x)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + self.eps).expand_as(out)        
        return out

def init_resnet101gem():
    model_ft = models.resnet101(pretrained=True)
    features = list(model_ft.children())[:-2]
    features.append(GeM())
    features.append(Whiten())
    model = nn.Sequential(*features)
    
    return model
    
    
    
    
    
    
# def load_densenet(model_path, num_classes):
#     model_ft = models.densenet169(pretrained=True)
#     num_ftrs = model_ft.classifier.in_features
#     model_ft.classifier = nn.Linear(num_ftrs, num_classes)

#     features = list(model_ft.children())[:-1]
#     features.append(nn.ReLU(inplace=True))
#     features.append(nn.AdaptiveAvgPool2d((1, 1)))
#     features.append(list(model_ft.children())[-1])

#     model = nn.Sequential(*features)
#     model_dict = model.state_dict()
#     load_dict = torch.load(model_path)

#     temp_dict = dict(load_dict)
#     for k, v in temp_dict.items():
#         s = list(k)
#         if s[:17] == list("module.classifier"):
#             del s[:17]
#             s.insert(0, "3")
#             newk = "".join(s)
#             load_dict[newk] = load_dict.pop(k)
#         else:
#             del s[:15]
#             s.insert(0, "0") 
#             newk = "".join(s)
#             load_dict[newk] = load_dict.pop(k)

#     model_dict = {k: v for k, v in load_dict.items() if k in model_dict}
#     model.load_state_dict(model_dict)

#     return model
    


