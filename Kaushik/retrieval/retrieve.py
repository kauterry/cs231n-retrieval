from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import faiss
import pickle

from model import load_model
from dataloader import load_data
from index_loader import load_index, index_read


def save_index(model, output_dim, dataloaders_dict, save_path, batch_size = 32, index_set = 'index'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    cpuIndex = faiss.IndexFlatL2(output_dim)

    data_index = []
    
    index_set_size = len(dataloaders_dict[index_set].dataset)
    print (index_set_size)
    
    for i, (inputs, labels, img_id) in enumerate(dataloaders_dict[index_set]):
        inputs = inputs.to(device)
        
        # forward
        print('batch %d / %d' % (i, index_set_size//batch_size))
        outputs = model(inputs).view(-1, output_dim).cpu().numpy()
        cpuIndex.add(outputs)
        data_index.extend(list(img_id))

#         if i >= 10:
#             break

    save_index = save_path + '.index'

    faiss.write_index(cpuIndex, save_index)

    data_path = save_path + '.txt'
    
    with open(data_path, "wb") as fp: 
        pickle.dump(data_index, fp)


def retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = 32, test_set = 'test'):
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_set_size = len(dataloaders_dict[test_set].dataset)
    print (test_set_size)
    
    save_index = save_path + '.index'

    cpuIndex = index_read(save_index)
    
    testIndex = faiss.IndexFlatL2(output_dim)
    
    index_feature = load_index(cpuIndex)
    print ("Index loaded.")
    
    data_path = save_path + '.txt'
    
    with open(data_path, "rb") as fp: 
        data_index = pickle.load(fp)
    print ("Data_Index loaded.")
    print (len(data_index))
    
    k = 100   
    sub = save_path + '.csv'

    last_batch_size_test = batch_size
    if (test_set_size % batch_size != 0):
        last_batch_size_test = test_set_size - (test_set_size//batch_size) * batch_size

    with open(sub, 'ab') as f:
        np.savetxt(f, [['id', 'images']], delimiter = ',',fmt="%s")
        for i, (inputs, labels, img_id) in enumerate(dataloaders_dict[test_set]):
            inputs = inputs.to(device)

            # forward   
            print('batch %d / %d' % (i, test_set_size//batch_size))
            with torch.set_grad_enabled(False):
                outputs = model(inputs).view(-1, output_dim).cpu().numpy()
                testIndex.add(outputs)
                
                D, I = index_feature.search(outputs, k)

                if (i == test_set_size//batch_size):
                    for b in range(last_batch_size_test):
                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")
                else:
                    for b in range(batch_size):
                        np.savetxt(f, [img_id[b]], newline = ',', fmt="%s")
                        indices = I[b]
                        neighbors = [[data_index[ind] for ind in indices]]
                        np.savetxt(f, neighbors, delimiter = ' ', fmt="%s")

#             if i >= 10:
#                 break

    save_index = save_path + 'test.index'
    faiss.write_index(testIndex, save_index)
            
if __name__ == '__main__':

    model_path = '/home/kylecshan/saved/06_01_2019/model1_epoch6.pth'
    save_path = '/home/kylecshan/saved/06_01_2019/resnetgem101'
    model_name = "resnet101gem"
    output_dim = 2048
    batch_size = 256
    
    dataloaders_dict = load_data(batch_size = batch_size)
    model = load_model(model_name, model_path)
    save_index(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    retrieve(model, output_dim, dataloaders_dict, save_path, batch_size = batch_size)
    
    
    
