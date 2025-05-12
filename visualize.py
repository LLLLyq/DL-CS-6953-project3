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
import json


# Visualize 3 to 5 test cases where the original model no longer classifies as expected
def visualization(dataset, adv_images, adv_labels, model, Config, epsilon):
    mean_norms = Config.mean_norms
    std_norms = Config.std_norms
    device = Config.device
    unnorm = transforms.Normalize((-mean_norms / std_norms).tolist(), (1.0 / std_norms).tolist())

    fig, ax = plt.subplots(5, 3, figsize=(12, 20))
    shown = 0
    for i in range(len(adv_images)):
        if shown >= 5:
            break
        original = dataset[i][0].unsqueeze(0).to(device)
        adv = adv_images[i]
        true_label = adv_labels[i]

        orig_pred = model(original).argmax(dim=1).item()
        adv_pred = model(adv).argmax(dim=1).item()
        if orig_pred != adv_pred:  # & (orig_pred == true_label)
            adv_img_show = unnorm(adv.squeeze().cpu())
            ori_img_show = unnorm(original.squeeze().cpu())
            noise = (adv_img_show-ori_img_show) #  * 10
            # noise = unnorm(adv.squeeze().cpu() - original.squeeze().cpu())
            # print(noise)
            ax[shown][0].imshow(transforms.ToPILImage()(ori_img_show))
            ax[shown][1].imshow(transforms.ToPILImage()(adv_img_show))
            ax[shown][2].imshow(transforms.ToPILImage()(noise))
            ax[shown][0].set_title(f"Original: class {orig_pred}", fontsize=15)
            ax[shown][1].set_title(f"Adversarial: class {adv_pred}", fontsize=15)
            ax[shown][2].set_title("Noise", fontsize=15)
            # ax[shown][0].set_title(f"original: {keys[orig_pred-401]}")
            # ax[shown][1].set_title(f"adversarial: {keys[adv_pred-401]}")
            ax[shown][0].axis('off')
            ax[shown][1].axis('off')
            ax[shown][2].axis('off')
            shown += 1
    if Config.patch:
        plt.savefig(f'images comparison/imgs_not_classified_as_expected - {Config.attack} - eps={str(epsilon)} patch.png')
    elif Config.after_adv:
        plt.savefig(f'images comparison/imgs_not_classified_as_expected - {Config.attack} - eps={str(epsilon)} after_adv.png')
    else:
        plt.savefig(f'images comparison/imgs_not_classified_as_expected - {Config.attack} - eps={str(epsilon)}.png')