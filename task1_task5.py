import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm

class Config:
    patch = False  # always False
    after_adv = True
    if after_adv:  # task 5
        dataset_path = ["./TestDataSet", './Adversarial_Test_Set_1', './Adversarial_Test_Set_2 PGD', './Adversarial_Test_Set_3 PGD']  # "./TestDataSet", './Adversarial_Test_Set_1', './Adversarial_Test_Set_2 PGD', './Adversarial_Test_Set_3 PGD - target attack'
        # model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
        # model = [torchvision.models.densenet121(weights='IMAGENET1K_V1'), torchvision.models.vgg16(weights='IMAGENET1K_V1'), torchvision.models.resnet50(weights='IMAGENET1K_V1')]  # torchvision.models.densenet121(weights='IMAGENET1K_V1'), torchvision.models.vgg16(weights='IMAGENET1K_V1'), torchvision.models.resnet50(weights='IMAGENET1K_V1')
        model = [torchvision.models.resnet34(weights='IMAGENET1K_V1')]
    else:  # task 1
        dataset_path = ["./TestDataSet"]
        model = [torchvision.models.resnet34(weights='IMAGENET1K_V1')]


def load_dataset(dataset_path, batch_size):
    # Load label mapping
    with open("./TestDataSet/labels_list.json", "r") as f:
        label_map = json.load(f)
    # Build index-to-class mapping from label_map
    name_to_idx = {}
    for entry in label_map:
        idx_str, class_name = entry.split(": ")
        name_to_idx[class_name.strip()] = int(idx_str.strip())  #

    # Preprocessing
    mean_norms = np.array([0.485, 0.456, 0.406])
    std_norms = np.array([0.229, 0.224, 0.225])

    plain_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norms.tolist(),
                             std=std_norms.tolist())
    ])

    def image_loader(path):
        return Image.open(path).convert('RGB')

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=plain_transforms)
    # mapping from dataset.class_to_idx (local) to ImageNet indices
    # dataset_to_imagenet_idx = {local_idx: name_to_idx[class_name] for class_name, local_idx in dataset.class_to_idx.items()}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return dataset, dataloader # , dataset_to_imagenet_idx


def load_model(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
    # model = Config.model
    model.eval().to(device)

    # Evaluate the model and report top-1 and top-5 accuracy
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)

            # The index in the .json file start from 401, while the predicted class label starts from 0.
            # Therefore, we add 401 to the class label.
            targets = targets.to(device) + 401
            # print('targets', targets)

            outputs = model(images)  # probabilities of different classes
            _, top1_pred = outputs.topk(1, dim=1)  # the No. of the top 1 class
            _, top5_pred = outputs.topk(5, dim=1)  # the No. of the top 5 classes
            # print('top1_pred', np.array(top1_pred.cpu().detach()).tolist())

            top1_correct += (top1_pred.squeeze() == targets).sum().item()
            top5_correct += sum([t in top5 for t, top5 in zip(targets, top5_pred)])

            total += targets.size(0)

    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")


if __name__=='__main__':
    print('Config.after_adv', Config.after_adv)
    for Model in Config.model:
        for Dataset_path in Config.dataset_path:
            print(Dataset_path)
            Dataset, Data_loader = load_dataset(Dataset_path, batch_size=32)
            load_model(Data_loader, Model)
        print('\n')




