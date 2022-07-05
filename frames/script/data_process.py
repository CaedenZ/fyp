import os
import csv

import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from tqdm import tqdm

from util import segmentation

# data argumentation and transformation for train set
transforms_train = transforms.Compose([
    transforms.RandomOrder([
        transforms.RandomRotation((-360, 360)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()]),
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data transformation for dev/test set
transforms_test = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dir = 'data/'
test_dir = 'data/'
seedling_list = (os.listdir(train_dir))
# ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
# 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
num2seedling_dict = dict(zip([i for i in range(12)], seedling_list))


def load_data() -> dict:
    """ load data from dataset and apply transformation (data augmentation)"""

    # read images
    image_datasets = datasets.ImageFolder(os.path.join(train_dir))
    # split the data into train and development sets
    # we have total 4750 images for training so split them to train and development sets approximately 85:15
    train_images, dev_images = random_split(image_datasets, [4000, 750])

    class MapDataSet(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __getitem__(self, index):
            x = segmentation(self.dataset[index][0])
            image = self.transform(x)
            label = self.dataset[index][1]
            return image, label

        def __len__(self):
            return len(self.dataset)

    # link image with its label and apply transforms
    train_set = MapDataSet(train_images, transforms_train)
    dev_set = MapDataSet(dev_images, transforms_test)

    return {'train': train_set, 'dev': dev_set}


def write_data(net: torch.nn.Module, file_name: str = 'result.csv', seg: bool = True) -> None:
    """write the prediction result to the csv file"""

    class TestDataSet(Dataset):
        def __init__(self, main_dir, transform):
            self.main_dir = main_dir
            self.transform = transform
            self.all_imgs = os.listdir(main_dir)

        def __len__(self):
            return len(self.all_imgs)

        def __getitem__(self, idx):
            label = self.all_imgs[idx]
            img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
            if seg is True:
                image = segmentation(Image.open(img_loc).convert("RGB"))
            else:
                image = Image.open(img_loc).convert("RGB")
            image = self.transform(image)
            return image, label

    # link image with its label(name) and apply transforms
    test_set = TestDataSet(test_dir, transforms_test)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, pin_memory=True)
    # read the names in the sample_submission.csv
    names = []
    with open("plant-seedlings-classification/sample_submission.csv", "r") as data_file:
        reader = csv.reader(data_file, delimiter=',')
        for row in reader:
            names.append(row[0])
    names.remove('file')
    # get the prediction result
    result_dict = {}
    net.eval()
    device = torch.device("cuda")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            result_dict[labels[0]] = predicted.item()
    # store the result as nested list for the csv writer
    result = [["file", "species"]]
    for name in names:
        result.append([name, num2seedling_dict[result_dict[name]]])
    # write the result.csv
    with open("%s" % file_name, "w", newline='\n') as result_file:
        writer = csv.writer(result_file, delimiter=',')
        writer.writerows(result)
    print("finish writing!")
