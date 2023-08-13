import numpy as np
import os
import torch
import argparse
import random
import math
from PIL import Image
import urllib.request
from torchvision import transforms
from modules import UNet_conditional, EMA
import ddpm_conditional

def get_free_gpu_idx():
    """
    returns index of the graphics card with the most free memory.
    If there are several graphics cards with the same amount of free memory, the one with the smaller index will be used.
    """
    maximum = 0
    id_graka = None
    for i in range(4):
        print(f"GPU {i} has {round(torch.cuda.mem_get_info(device=i)[0]/(1024**3), 3)}GB free memory out of {round(torch.cuda.mem_get_info(device=i)[1]/(1024**3), 3)}GB")
        if maximum < torch.cuda.mem_get_info(device=i)[0]:
            maximum = torch.cuda.mem_get_info(device=i)[0]
            id_graka = i
    return id_graka


def generate_images(model_epoch, size, device, batch, random_seed, generative_model_path, images_path):
    """
    model_epoch = every generativ model is labelled with the number of epochs its trained. This variable indicates which model will be loaded
    size = size for the dataset, number of images which will be generated
    device = gives the index of the graphics card which will be used for generating the data. Can be set to None.
    batch = Number of images which will be generated at once
    random_seed = random seed
    generative_model_path = path to the folder, where the generative model is saved
    images_path = path where the dataset will be saved
    """
    if not os.path.exists(f"{images_path}"):
        os.makedirs(f"{images_path}")
    
    if device is None:
        device = get_free_gpu_idx()
    assert size % 10 == 0
    
    # airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9
    n = 10
    size_per_class = int(size/10)
    
    assert size_per_class % batch == 0, "Batch must be a divisor of the number of images to be generated."

    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load(f"{generative_model_path}/ema_ckpt_{model_epoch}.pt")
    model.load_state_dict(ckpt)
    diffusion = ddpm_conditional.Diffusion(img_size=32, device=device)
    
    try:
        loaded_data = np.load(f"{images_path}/data_epoch_{model_epoch}_{random_seed}.npz")
        x = loaded_data["x"]
        y_from_data = loaded_data["y"]
        counter = loaded_data["counter"]
        was_shuffled = loaded_data["was_shuffled"]
        assert sum(counter) == len(x) == len(y_from_data)
    except:
        counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        was_shuffled = False
    
    assert (size - sum(counter)) % batch == 0, f"Batch must be a divisor of the remaining number of images to be generated. Remaining number of images to be generated is {size - sum(counter)}"
    
    for target in range(n):
        while counter[target] < size_per_class:
            print(f"generate image from class: {target}")
            print(f"counter: {counter}")
            if target == 0 and sum(counter) == 0:
                y = torch.Tensor([target] * batch).long().to(device)
                with torch.no_grad():
                    x = diffusion.sample(model, batch, y, cfg_scale=3).cpu()
                counter[target] += batch
                y_from_data = np.array([target for class_number_itt in range(batch)])
                np.savez(f"{images_path}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
            else:
                y = torch.Tensor([target] * batch).long().to(device)
                with torch.no_grad():
                    x = np.concatenate((x, diffusion.sample(model, batch, y, cfg_scale=3).cpu()))
                counter[target] += batch
                y_from_data = np.concatenate((y_from_data, np.array([target for class_number_itt in range(batch)])))
                np.savez(f"{images_path}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
    
    print(f"counter: {counter}")
    
    if not was_shuffled:
        print("shuffle Data")
        combined_data = [(x_i,y_i) for x_i,y_i in zip(x,y_from_data)]
        random.shuffle(combined_data)
        x = np.array([x_i[0] for x_i in combined_data])
        y_from_data = np.array([y_i[1] for y_i in combined_data])
        was_shuffled = True
    
        # wird beim erzeugen als (size, 3, 32, 32) und muss zu 
        # (size, 32, 32, 3) verändert werden (Format von cifar5m)

        while x.shape != (size, 32, 32, 3):
            print("transpose data")
            x = np.transpose(x, (0, 2, 3, 1))
        print(f"data shape: {x.shape} == ({size}, 32, 32, 3)")

        assert sum(counter) == len(x) == len(y_from_data)
        np.savez(f"{images_path}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
    
    return x, y_from_data

