import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from task1_task5 import load_dataset
from visualize import visualization
import json

class Config:
    mean_norms = np.array([0.485, 0.456, 0.406])
    std_norms = np.array([0.229, 0.224, 0.225])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = [0.005, 0.01, 0.02]  # attack budget:  If raw (unpreprocessed) images have pixel values of 0-255, an attack budget of ε = 0.02 roughly corresponds to changing each pixel value in the raw image by at most +/-1.
    attack = 'FGSM attack'
    patch = False
    after_adv = False

# Fast Gradient Sign Method (FGSM)
def fgsm_attack(image, label, model, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()
    grad = image.grad.data
    perturbed = image + epsilon * grad.sign()
    return perturbed


# Implement FGSM for each image in the test dataset for ε = 0.02
def attack_run(dataloader, epsilon):
    mean_norms = Config.mean_norms
    std_norms = Config.std_norms
    device = Config.device
    eps = epsilon

    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model.eval().to(device)

    adv_images = []
    adv_labels = []
    original_images = []
    original_labels = []
    num_changed = 0
    os.makedirs("Adversarial_Test_Set_1", exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(dataloader)):
        img = img.to(device)
        true_label = label.to(device)+401

        output = model(img)
        pred = output.argmax(dim=1)

        eps_norm = eps / torch.tensor(std_norms, dtype=torch.float).reshape((3, 1, 1))
        adv_img = fgsm_attack(img, true_label, model, eps_norm.to(device))  # use FGSM to generate adversarial images. Attack is applied in normalized units
        adv_output = model(adv_img)
        adv_pred = adv_output.argmax(dim=1)

        if adv_pred != true_label:
            num_changed += 1

        # Save the adversarial image (un-normalized for visualization)
        unnorm = transforms.Normalize(
            mean=(-mean_norms / std_norms).tolist(),
            std=(1.0 / std_norms).tolist()
        )
        img_to_save = unnorm(adv_img.squeeze().cpu())
        img_pil = transforms.ToPILImage()(img_to_save)
        adv_images.append(adv_img)
        adv_labels.append(true_label)

        true_class = int(label.item()+401)  # e.g., 401, 402, etc.
        folder_name = str(true_class)
        os.makedirs(f"Adversarial_Test_Set_1/{folder_name}", exist_ok=True)
        img_pil.save(f"Adversarial_Test_Set_1/{folder_name}/adv_img_{idx:4d}.png")
        # original_images.apped(img)
        # original_labels.append(true_label)
    print('num_changed', num_changed)
    return adv_images, adv_labels, model  # original_images, original_labels,


def evaluate_new_set(adv_images, adv_labels, model):
    device = Config.device
    top1, top5, total = 0, 0, 0
    for i in range(len(adv_images)):
        img = adv_images[i].to(device)
        label = adv_labels[i]

        output = model(img)
        _, top1_pred = output.topk(1, dim=1)
        _, top5_pred = output.topk(5, dim=1)

        top1 += (top1_pred.squeeze() == label).sum().item()
        top5 += sum([label.item() in top5_pred[j] for j in range(top5_pred.size(0))])
        total += 1

    top1_acc = 100 * top1 / total
    top5_acc = 100 * top5 / total
    print(f"[Adversarial] Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"[Adversarial] Top-5 Accuracy: {top5_acc:.2f}%")


if __name__=='__main__':
    Dataset_path = "./TestDataSet"
    Dataset, Data_loader = load_dataset(Dataset_path, batch_size=1)
    for Eps in Config.epsilon:
        print(f'epsilon: {Eps}')
        Adv_images, Adv_labels, Model = attack_run(Data_loader, Eps)  # Original_images, Original_labels,
        visualization(Dataset, Adv_images, Adv_labels, Model, Config, Eps)
        evaluate_new_set(Adv_images, Adv_labels, Model)











