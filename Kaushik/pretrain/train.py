from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np

from training import train_model
from model import initialize_model


def train(model_name, freeze_layers, model_number, optimizer, learning_rate, num_classes = 2000, batch_size = 32, num_epochs = 10, input_size = 224, data_dir = "/home/kylecshan/data/images224/train_ms2000_v5/", save_path = "/home/kylecshan/saved/06_01_2019/classification/delf/"):

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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
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
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    save_every = 1
    for i in range(0, num_epochs, save_every):
        model_ft, val_hist, train_hist = train_model(model_ft, dataloaders_dict, criterion,
                                     optimizer_ft, device, num_epochs=save_every,
                                     is_inception=(model_name=="inception"))

        # Save the model parameters as a .pth file
        save_dir = save_path + "model" + str(model_number) + "_epoch"+ str(i+save_every-1) + ".pth"
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
    model_name = ["delf"]

    # Number of layers to freeze. If freeze_layers == 9, we freeze all layers 
    # except the fully connected layer (For ResNet50)
    freeze_layers = [0]
    optimizers = ["Adam"]
    learning_rates = [1e-3]
    
    model_number = 1
    for mod_name, f_layers, optims, lr in zip(model_name, freeze_layers, optimizers, learning_rates):
        train(mod_name, f_layers, model_number, optims, lr)
        model_number += 1
