from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import random

from model import load_model
from dataloader import load_data
from index_loader import load_index, index_read


def save_index(model, output_dim, dataloaders_dict, save_path, batch_size = 32, index_set = 'index'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    data_index = []
    label_index = []
    preds_index = []
    
    index_set_size = len(dataloaders_dict[index_set].dataset)
    print (index_set_size)
    
    for i, (inputs, labels, img_id) in enumerate(dataloaders_dict[index_set]):
        inputs = inputs.to(device)
        
        # forward
        print('batch %d / %d' % (i, index_set_size//batch_size))
        outputs = model(inputs).view(-1, output_dim)
        _, preds = torch.max(outputs, 1)
        preds_index.extend(preds.cpu().tolist())
        data_index.extend(list(img_id))
        label_index.extend(labels.cpu().tolist())

#         if i >= 2:
#             break

    data_path = save_path + '.txt'
    
    with open(data_path, "wb") as fp: 
        pickle.dump([preds_index, data_index, label_index], fp)


def retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = 32, test_set = 'test'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_set_size = len(dataloaders_dict[test_set].dataset)
    print (test_set_size)
    
    data_path = save_path + '.txt'
    
    with open(data_path, "rb") as fp: 
        preds_index, data_index, label_index = pickle.load(fp)
    print ("All indexes loaded.")
    preds_index = np.array(preds_index)
    print (np.mean(preds_index == np.array(label_index)))
    k = 100   
    sub = save_path + 'idx.csv'
    class_file = save_path + 'label.csv'

    last_batch_size_test = batch_size
    if (test_set_size % batch_size != 0):
        last_batch_size_test = test_set_size - (test_set_size//batch_size) * batch_size

    with open(sub, 'ab') as f, open(class_file, 'ab') as g:
        np.savetxt(f, [['id', 'images']], delimiter = ',',fmt="%s")
        for i, (inputs, labels, img_id) in enumerate(dataloaders_dict[test_set]):
            inputs = inputs.to(device)

            # forward   
            print('batch %d / %d' % (i, test_set_size//batch_size))
            with torch.set_grad_enabled(False):
                outputs = model(inputs).view(-1, output_dim)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()

                if (i == test_set_size//batch_size):
                    for b in range(last_batch_size_test):
                        t_label = labels[b].cpu().numpy()
                        np.savetxt(g, [t_label], newline = ',', fmt="%s")
                        correct_indices = np.where(preds_index == preds[b])[0][:k]
                        if len(correct_indices) < k and len(correct_indices) > 0:
                            rand_indices = np.random.choice([x for x in range(preds_index.shape[0]) if x not in correct_indices], k - len(correct_indices), False)
                            indices = np.concatenate((correct_indices, rand_indices))
                        elif len(correct_indices) == 0:
                            indices = np.random.choice([x for x in range(preds_index.shape[0]) if x not in correct_indices], k - len(correct_indices), False)
                        else:
                            indices = correct_indices

                        neighbors_labels = [[label_index[ind] for ind in indices]]
                        np.savetxt(g, neighbors_labels, delimiter = ',', fmt="%s")

                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")
                else:
                    for b in range(batch_size):
                        t_label = labels[b].cpu().numpy()
                        np.savetxt(g, [t_label], newline = ',', fmt="%s")
                        correct_indices = np.where(preds_index == preds[b])[0][:k]
                        if len(correct_indices) < k and len(correct_indices) > 0:
                            rand_indices = np.random.choice([x for x in range(preds_index.shape[0]) if x not in correct_indices], k - len(correct_indices), False)
                            indices = np.concatenate((correct_indices, rand_indices))
                        elif len(correct_indices) == 0:
                            indices = np.random.choice([x for x in range(preds_index.shape[0]) if x not in correct_indices], k - len(correct_indices), False)
                        else:
                            indices = correct_indices
                        neighbors_labels = [[label_index[ind] for ind in indices]]
                        np.savetxt(g, neighbors_labels, delimiter = ',', fmt="%s")

                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")
#             if i >= 10:
#                 break


            
if __name__ == '__main__':

    model_path = '/home/kylecshan/saved/06_01_2019/classification/densenet169/model1_epoch9.pth'
    save_path = '/home/kylecshan/saved/06_01_2019/classification/densenet169/densenetclass/densenetclass'
    model_name = "densenet_class"
    output_dim = 2000
    batch_size = 256
    
    dataloaders_dict = load_data(batch_size = batch_size)
    model = load_model(model_name, model_path)
#     save_index(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    
    
    
