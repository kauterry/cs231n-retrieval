from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import faiss
import pickle
from scipy.spatial import distance

from model import load_model, initialize_model
from dataloader import load_data
from index_loader import load_index, index_read


def save_index(model, output_dim, dataloaders_dict, save_path, batch_size = 32, index_set = 'index'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    cpuIndex = []
    
    data_index = []
    label_index = []
    
    index_set_size = len(dataloaders_dict[index_set].dataset)
    print (index_set_size)
    
    for i, (inputs, labels, img_id) in enumerate(dataloaders_dict[index_set]):
        inputs = inputs.to(device)
        
        # forward
        print('batch %d / %d' % (i, index_set_size//batch_size))
        outputs = model(inputs).view(-1, output_dim).cpu().tolist()
        cpuIndex.extend(outputs)
        data_index.extend(list(img_id))
        label_index.extend(labels.cpu().tolist())

#         if i >= 10:
#             break

    data_path = save_path + '.txt'
    
    with open(data_path, "wb") as fp: 
        pickle.dump([cpuIndex, data_index, label_index], fp)


def retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = 32, test_set = 'test'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_set_size = len(dataloaders_dict[test_set].dataset)
    print (test_set_size)
        
    data_path = save_path + '.txt'
    
    with open(data_path, "rb") as fp: 
        index_feature, data_index, label_index = pickle.load(fp)
    print ("Data_Index, Label Index loaded.")
    print (len(data_index), len(label_index))
    index_feature = np.array(index_feature)
    
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
                outputs = model(inputs).view(-1, output_dim).cpu().numpy()
                
                D = distance.cdist(outputs, index_feature, 'euclidean')
                I = np.argsort(D, axis = 1)[:,:k]

                if (i == test_set_size//batch_size):
                    for b in range(last_batch_size_test):
                        t_label = labels[b].cpu().numpy()
                        np.savetxt(g, [t_label], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors_labels = [[label_index[ind] for ind in indices]]
                        np.savetxt(g, neighbors_labels, delimiter = ',', fmt="%s")

                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")
                else:
                    for b in range(batch_size):
                        t_label = labels[b].cpu().numpy()
                        np.savetxt(g, [t_label], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors_labels = [[label_index[ind] for ind in indices]]
                        np.savetxt(g, neighbors_labels, delimiter = ',', fmt="%s")

                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")
#             if i >= 10:
#                 break
            
if __name__ == '__main__':

#     model_path = '/home/kylecshan/saved/06_01_2019/triplet/delf_pca768/model1_epoch0.pth'
    save_path = '/home/kylecshan/saved/06_04_2019/densenet_base_no_faiss/densenet_base'
    model_name = "densenet_class"
    output_dim = 1664
    batch_size = 256
    
    dataloaders_dict = load_data(batch_size = batch_size)
#     model = load_model(model_name, model_path)
    model = initialize_model(model_name, use_pretrained=True)
    save_index(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    
    
    
