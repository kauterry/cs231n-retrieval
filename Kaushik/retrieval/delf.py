import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, utils, models


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

        
def delf_densenet169():
   
    model_ft = models.densenet169(pretrained=True)
    
    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(AttentionPool(1664))
#     features.append(list(model_ft.children())[-1])
    delf_model = nn.Sequential(*features)
    
    return delf_model

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

def init_delf_pca():

    model_ft = models.densenet169(pretrained=True)
    features = list(model_ft.children())[:-1]
    features.append(nn.ReLU(inplace=True))
    features.append(AttentionPool(1664))
    model = nn.Sequential(*features)
    model = nn.DataParallel(model)
    model_pca = nn.Sequential(*list(model.children()), nn.Linear(1664, 768))

    return model_pca   


