import os
import torch
from torchvision import datasets, transforms

# data_dir = '/home/kylecshan/data/stage2/'
# data_dir = '/home/kylecshan/data/images224/train_ms2000_v4/'

def load_data(batch_size = 32, input_size = 224, data_dir = "/home/kylecshan/data/images224/train_ms2000_v5/", index_set = 'index', test_set = 'test'):

    data_transforms = {
        index_set: transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        test_set: transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: ImageFolderWithID(os.path.join(data_dir, x), data_transforms[x]) for x in [index_set, test_set]}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle = False, num_workers=4) for x in [index_set, test_set]}
              
    return dataloaders_dict


class ImageFolderWithID(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithID, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        base = os.path.basename(path)
        img_id = os.path.splitext(base)[0]
        # make a new tuple that includes original and the image id
        tuple_with_path = (original_tuple + (img_id,))
        return tuple_with_path

    
    