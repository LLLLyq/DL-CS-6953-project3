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
import random

seed = 42  # or any integer you like
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Config:
    mean_norms = np.array([0.485, 0.456, 0.406])
    std_norms = np.array([0.229, 0.224, 0.225])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = 0.005
    iters = 10
    attack = 'PGD'  # 'PGD - target attack'  # 'PGD'
    patch = True
    if patch:  #  task4: 'PGD - target attack'
        epsilon = [0.02, 0.3, 0.5]  # attack budget
    else:  # task3: 'PGD'
        # if not pathc, eps should not greater than 0.02
        epsilon = [0.02]  # attack budget
    after_adv = False  # always False
    patch_size = 32


# PGD (Projected Gradient Descent): multi-step attacks
def pgd_attack(image, label, model, eps, alpha=Config.alpha, iters=Config.iters):
    ori_image = image.clone().detach()
    perturbed = image.clone().detach()
    perturbed.requires_grad = True

    _, _, h, w = image.shape
    patch_size = Config.patch_size
    x0 = random.randint(0, w - patch_size)  # random seed is needed 
    y0 = random.randint(0, h - patch_size)

    for _ in range(iters):
        output = model(perturbed)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data
        if Config.patch:
            patch_grad = grad[:, :, y0:y0 + patch_size, x0:x0 + patch_size].sign()
            patch = perturbed[:, :, y0:y0 + patch_size, x0:x0 + patch_size] - alpha * patch_grad
            perturbation = patch + ori_image[:, :, y0:y0 + patch_size, x0:x0 + patch_size]
            perturbation = torch.clamp(perturbation, -eps, eps)
            # Update patch
            perturbed = perturbed.clone().detach()
            perturbed[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = (ori_image[:, :, y0:y0 + patch_size, x0:x0 + patch_size] + perturbation)
            perturbed = perturbed.detach()
            perturbed.requires_grad = True
        else:  # not patch
            perturbed = perturbed + alpha * grad.sign()
            # Project back to epsilon L-inf ball
            perturbation = torch.clamp(perturbed - ori_image, min=-eps, max=eps)
            # perturbed = torch.clamp(ori_image + perturbation, min=0, max=1).detach()
            perturbed = (ori_image + perturbation).detach()
            perturbed.requires_grad = True

    return perturbed  # torch.clamp(perturbed, 0, 1)


def pgd_target_attack(image, label, model, eps, alpha=Config.alpha, iters=Config.iters):
    ori_image = image.clone().detach()
    perturbed = image.clone().detach()
    perturbed.requires_grad = True

    _, _, h, w = image.shape
    patch_size = Config.patch_size
    x0 = random.randint(0, w - patch_size)
    y0 = random.randint(0, h - patch_size)

    for _ in range(iters):
        output = model(perturbed)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed.grad.data
        if Config.patch:
            patch_grad = grad[:, :, y0:y0 + patch_size, x0:x0 + patch_size].sign()
            patch = perturbed[:, :, y0:y0 + patch_size, x0:x0 + patch_size] - alpha * patch_grad  # for targeted attack, we use negative gradient
            perturbation = patch - ori_image[:, :, y0:y0 + patch_size, x0:x0 + patch_size]
            perturbation = torch.clamp(perturbation, -eps, eps)
            # Update patch
            perturbed = perturbed.clone().detach()
            perturbed[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = (ori_image[:, :, y0:y0 + patch_size, x0:x0 + patch_size] + perturbation)
            # perturbed = torch.clamp(perturbed, 0, 1).detach()
            perturbed = perturbed.detach()
            perturbed.requires_grad = True
        else:  # not patch
            perturbed = perturbed - alpha * grad.sign()  # for targeted attack, we use negative gradient
            # Project back to epsilon L-inf ball
            perturbation = torch.clamp(perturbed - ori_image, min=-eps, max=eps)
            # perturbed = torch.clamp(ori_image + perturbation, min=0, max=1).detach()
            perturbed = (ori_image + perturbation).detach()
            perturbed.requires_grad = True

    return perturbed  # torch.clamp(perturbed, 0, 1)


# Implement FGSM for each image in the test dataset for ε = 0.02
def attack_run(dataloader, epsilon):
    print('patch', Config.patch)
    mean_norms = Config.mean_norms
    std_norms = Config.std_norms
    device = Config.device
    eps = epsilon

    model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    model.eval().to(device)

    adv_images = []
    adv_labels = []
    num_changed = 0
    if Config.patch:
        print('patch')
        os.makedirs(f"Adversarial_Test_Set_3 {Config.attack}", exist_ok=True)
    else:
        print(' no patch')
        os.makedirs(f"Adversarial_Test_Set_2 {Config.attack}", exist_ok=True)

    for idx, (img, label) in enumerate(tqdm(dataloader)):
        img = img.to(device)
        true_label = label.to(device)+401

        output = model(img)
        pred = output.argmax(dim=1)

        eps_norm = eps / torch.tensor(std_norms, dtype=torch.float).reshape((3, 1, 1))
        if Config.attack=='PGD':
            adv_img = pgd_attack(img, true_label, model, eps_norm.to(device))  # use PGD to generate adversarial images. Attack is applied in normalized units
        elif Config.attack=='PGD - target attack':
            target_class = torch.randperm(true_label.size(0), device='cuda') #+401  # partly overlap with the range of original labels
            # target_class = random.choice([i for i in range(1000) if i != true_label.item()])  # do not overlap with the range of original labels
            target_label = torch.tensor([target_class], device=device)
            adv_img = pgd_target_attack(img, target_label, model, eps_norm.to(device))
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

        true_class = int(label.item() + 401)  # e.g., 401, 402, etc.
        folder_name = str(true_class)
        if Config.patch:
            os.makedirs(f"Adversarial_Test_Set_3 {Config.attack}/{folder_name}", exist_ok=True)
            img_pil.save(f"Adversarial_Test_Set_3 {Config.attack}/{folder_name}/adv_img_{idx:4d}.png")
            # print(f"Adversarial_Test_Set_3 {Config.attack}")
        else:
            os.makedirs(f"Adversarial_Test_Set_2 {Config.attack}/{folder_name}", exist_ok=True)
            img_pil.save(f"Adversarial_Test_Set_2 {Config.attack}/{folder_name}/adv_img_{idx:4d}.png")
            # print(f"Adversarial_Test_Set_2 {Config.attack}")
    print('num_changed', num_changed)
    return adv_images, adv_labels, model


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
    print(Config.attack)
    print(f"[Adversarial] Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"[Adversarial] Top-5 Accuracy: {top5_acc:.2f}%")


if __name__=='__main__':
    Dataset_path = "./TestDataSet"
    Dataset, Data_loader = load_dataset(Dataset_path, batch_size=1)
    for Eps in Config.epsilon:
        print(f'\nepsilon: {Eps}')
        Adv_images, Adv_labels, Model = attack_run(Data_loader, Eps)
        visualization(Dataset, Adv_images, Adv_labels, Model, Config, Eps)
        evaluate_new_set(Adv_images, Adv_labels, Model)
    '''
    print(Config.patch)
    print(~Config.patch)
    print((Config.patch))
    '''











