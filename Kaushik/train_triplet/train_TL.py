from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os, sys
import numpy as np
import random

from training_TL import train_model_TL
sys.path.insert(0, "/home/kylecshan/Kaushik/pretrain/")
from model import initialize_model

# Adapted from https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18
class TripletDataset(torch.utils.data.Dataset):
    """Dataset that on each iteration provides an anchor image,
    a positive match, and a negative match, in that order.
    """
    
    
    def __init__(self, dataset):
        self.base_dataset = dataset
        
        #find index of each class
        class_indices = []
        image_classes = np.array([x[1] for x in self.base_dataset.imgs])
        num_classes = len(self.base_dataset.classes)
        for c in range(num_classes):
            indices = np.nonzero(image_classes == c)[0]
            class_indices.append(indices)

        #generate balanced pairs
        self.trip_indexes = np.zeros((len(self.base_dataset), 3), dtype=int)
        i = 0
        for c in range(num_classes):
            num_examples = len(class_indices[c])
            for j in range(num_examples):
                # choose another class at random
                neg_cls = random.randint(0, num_classes-2)
                if neg_cls >= c:
                    neg_cls += 1
                neg_idx = random.randint(0, len(class_indices[neg_cls])-1)
                
                # choose another picture of the same class at random
                pos_idx = random.randint(0, num_examples-2)
                if pos_idx >= j:
                    pos_idx += 1
                self.trip_indexes[i,:] = [class_indices[c][j],
                                          class_indices[c][pos_idx],
                                          class_indices[neg_cls][neg_idx]]
                i += 1
    def __getitem__(self,index):
        """ Output is a 3 x C x H x W tensor, and [1,0] indicating positive and negative match
        """
        idxs = self.trip_indexes[index]
        
        triplet = []
        for idx in idxs:
            triplet.append(self.base_dataset[idx][0])
                
        return torch.stack(triplet, dim=0), torch.tensor([1, 0])
    
    def __len__(self):
        return len(self.trip_indexes)
        
        
class triplet_loss:
    """ Input is model outputs (list of three N x D tensors, where N is batch size and
        D is dimension of image representation)
        Output is the triplet loss
    """
    def __init__(self, margin=0.1, norm=True):
        self.margin = margin
        self.norm = norm
        
    def __call__(self, output_triple):
        anc, pos, neg = output_triple
        if self.norm:
            # Normalize outputs
            for x in (anc, pos, neg):
                x = F.normalize(x, p=2, dim=1)
        # Compute squared l2 distance and loss
        dist_pos = torch.sum(torch.pow(anc - pos, 2), dim=1)
        dist_neg = torch.sum(torch.pow(anc - neg, 2), dim=1)
        loss = 0.5 * torch.mean((self.margin + dist_pos - dist_neg).clamp(min=0))
        return loss


def train(model_name, freeze_layers, model_number, optimizer, learning_rate, num_classes = 2000, batch_size = 256, num_epochs = 20, input_size = 224, data_dir = "/home/kylecshan/data/images224/train_ms2000_v5/", save_path = "/home/kylecshan/saved/06_01_2019/triplet/delf_pca768/"):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    for x in ['train', 'val']:
        image_datasets[x] = TripletDataset(image_datasets[x])

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4)
                        for x in ['train', 'val']}
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model for this run
    model_ft = initialize_model(model_name, num_classes, freeze_layers, use_pretrained=True)

    # Send the model to GPU
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    print("Params to learn:")
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    # Observe that all parameters are being optimized
    if optimizer == "Adam":
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    elif optimizer == "SGD":
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

    # Setup the loss fxn
    criterion = triplet_loss(margin=0.1, norm=True)

    # Train and evaluate
    save_every = 1
    for i in range(0, num_epochs, save_every):
        model_ft, val_hist, train_hist = train_model_TL(model_ft, dataloaders_dict, criterion,
                                     optimizer_ft, device, num_epochs=save_every,
                                     is_inception=(model_name=="inception"))

        # Save the model parameters as a .pth file
        save_dir = save_path + "model" + str(model_number) +"_epoch"+ str(i+save_every-1) + ".pth"
        torch.save(model_ft.state_dict(), save_dir)

        #Save Validation History (val_hist) to csv file
        save_val_csv = save_path + "val" + str(model_number) + ".csv"
        with open(save_val_csv, 'ab') as f_val:
            np.savetxt(f_val, val_hist)

        #Save Training History (train_hist) to csv file
        save_train_csv = save_path + "train" + str(model_number) + ".csv"
        with open(save_train_csv, 'ab') as f_train:
            np.savetxt(f_train, train_hist)

if __name__ == '__main__':

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = ["delf_pca"]

    # Number of layers to freeze. If freeze_layers == 9, we freeze all layers 
    # except the fully connected layer (For ResNet50)
    freeze_layers = [2]
    optimizers = ["Adam"]
    learning_rates = [1e-3]
    
    model_number = 1
    for mod_name, f_layers, optims, lr in zip(model_name, freeze_layers, optimizers, learning_rates):
        train(mod_name, f_layers, model_number, optims, lr, num_epochs = 10, batch_size = 32)
        model_number += 1
