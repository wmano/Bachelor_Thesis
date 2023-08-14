import numpy as np
import os
import torch
import argparse
import random
import math
from PIL import Image
import urllib.request
import module
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


def generate_images(model_epoch, size, device, batch, random_seed):
    assert size % 10 == 0
    
    # airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9
    n = 10
    size_per_class = int(size/10)
    
    assert size_per_class % batch == 0, "Batch must be a divisor of the number of images to be generated."

    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load(f"models/DDPM_conditional/ema_ckpt_{model_epoch}.pt")
    model.load_state_dict(ckpt)
    diffusion = ddpm_conditional.Diffusion(img_size=32, device=device)
    
    try:
        loaded_data = np.load(f"../saved_models/epoch_{model_epoch}/data_epoch_{model_epoch}_{random_seed}.npz")
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
                np.savez(f"../saved_models/epoch_{model_epoch}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
            else:
                y = torch.Tensor([target] * batch).long().to(device)
                with torch.no_grad():
                    x = np.concatenate((x, diffusion.sample(model, batch, y, cfg_scale=3).cpu()))
                counter[target] += batch
                y_from_data = np.concatenate((y_from_data, np.array([target for class_number_itt in range(batch)])))
                np.savez(f"../saved_models/epoch_{model_epoch}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
    
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
        np.savez(f"../saved_models/epoch_{model_epoch}/data_epoch_{model_epoch}_{random_seed}.npz", **{"x": x, "y": y_from_data, "counter": counter, "was_shuffled": was_shuffled})
    
    return x, y_from_data


def data_to_loaders(x, y):
    print("create dataloader")
    train_x = np.array(x[:45000])
    train_y = np.array(y[:45000])
    val_x = np.array(x[45000:50000])
    val_y = np.array(y[45000:50000])
    del x, y
    
    train_x = torch.from_numpy(train_x).type(torch.float).transpose(-1, 1)
    train_y = torch.from_numpy(train_y)
    val_x = torch.from_numpy(val_x).type(torch.float).transpose(-1, 1)
    val_y = torch.from_numpy(val_y)
    
    train_set = module.TTensorDataset([train_x, train_y])
    DATA_MEANS = (train_set[:][0] / 255.0).mean(axis=(0, 2, 3))
    DATA_STD = (train_set[:][0] / 255.0).std(axis=(0, 2, 3))
    
    test_transform = transforms.Compose([transforms.Normalize(DATA_MEANS, DATA_STD)])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )
    
    train_set = module.TTensorDataset([train_x, train_y], transform=train_transform)
    val_set = module.TTensorDataset([val_x, val_y], transform=test_transform)
    
    train_loader = module.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = module.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


def train_bootstrap(random_seed, model_epoch, size, batch):
    if not os.path.exists(f"../saved_models/epoch_{model_epoch}"):
        os.makedirs(f"../saved_models/epoch_{model_epoch}")
    gpu_idx = get_free_gpu_idx()
    num_workers = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.current_device() != gpu_idx:
        print('os.environ["CUDA_VISIBLE_DEVICES"] hat nicht geklappt')
        torch.cuda.set_device(gpu_idx)

    device = torch.cuda.current_device()
    print("PyTorch verwendet GPU:", device)
    
    x, y = generate_images(model_epoch, size, device, batch, random_seed)
    
    train_loader, val_loader = data_to_loaders(x, y)
    
    print("start training")
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", f"../saved_models/epoch_{model_epoch}")
    resnet_model, resnet_results = module.train_ensemble_model(
        model_name="ResNet",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=CHECKPOINT_PATH,
        model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        seed=random_seed,
        max_epochs=180,
        verbose=False
    )
    if not os.path.exists(f"../saved_models/epoch_{model_epoch}/resnet_results_seed_{random_seed}"):
        file = open(os.path.join(f"../saved_models/epoch_{model_epoch}/resnet_results_seed_{random_seed}"), "w")
        file.write(str(resnet_results))
    

def train_on_cifar5m(random_seed):
    gpu_idx = get_free_gpu_idx()
    num_workers = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.current_device() != gpu_idx:
        print('os.environ["CUDA_VISIBLE_DEVICES"] hat nicht geklappt')
        torch.cuda.set_device(gpu_idx)

    device = torch.cuda.current_device()
    print("PyTorch verwendet GPU:", device)
    
    train_loader, val_loader, test_loader = get_sets(random_seed)
    
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "../saved_models/CIFAR5m")
    resnet_model, resnet_results = module.train_model(
        model_name="ResNet",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        checkpoint_path=CHECKPOINT_PATH,
        model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        seed=random_seed,
        max_epochs=180,
        verbose=False
    )
    file = open(os.path.join(f"../saved_models/CIFAR5m/resnet_results_seed_{random_seed}"), "w")
    file.write(str(resnet_results))
    
    
def get_sets(random_seed):
    random.seed(random_seed)
    L = list(range(6))
    random.shuffle(L)
    draws_left = 60000
    for i in range(6):
        data_dict = np.load(f'../cifar5m_data/cifar5m_part{L[i]}.npz')
        if i != 5:
            ziehung = random.randint(0, draws_left)
            #print('ziehung', ziehung)
        else:
            ziehung = draws_left
        von = random.randint(0, 1000448-60000)
        #print('von', von)
        draws_left -= ziehung
        if i == 0:
            x = data_dict["X"][von:von+ziehung]
            y = data_dict["Y"][von:von+ziehung]
        else:
            x = np.concatenate((x, data_dict["X"][von:von+ziehung]))
            y = np.concatenate((y, data_dict["Y"][von:von+ziehung]))
        del data_dict
    train_x = x[:45000]
    train_y = y[:45000]
    val_x = x[45000:50000]
    val_y = y[45000:50000]
    test_x = x[50000:60000]
    test_y = y[50000:60000]
    del x, y
    
    train_x = torch.from_numpy(train_x).type(torch.float).transpose(-1, 1)
    train_y = torch.from_numpy(train_y)
    val_x = torch.from_numpy(val_x).type(torch.float).transpose(-1, 1)
    val_y = torch.from_numpy(val_y)
    test_x = torch.from_numpy(test_x).type(torch.float).transpose(-1, 1)
    test_y = torch.from_numpy(test_y)
    
    train_set = module.TTensorDataset([train_x, train_y])
    DATA_MEANS = (train_set[:][0] / 255.0).mean(axis=(0, 2, 3))
    DATA_STD = (train_set[:][0] / 255.0).std(axis=(0, 2, 3))
    
    test_transform = transforms.Compose([transforms.Normalize(DATA_MEANS, DATA_STD)])
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.Normalize(DATA_MEANS, DATA_STD),
        ]
    )
    
    train_set = module.TTensorDataset([train_x, train_y], transform=train_transform)
    val_set = module.TTensorDataset([val_x, val_y], transform=test_transform)
    test_set = module.TTensorDataset([test_x, test_y], transform=test_transform)
    
    train_loader = module.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = module.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False)
    test_loader = module.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader

def train_on_cifar5m_seed_7(random_seed):
    gpu_idx = get_free_gpu_idx()
    num_workers = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.current_device() != gpu_idx:
        print('os.environ["CUDA_VISIBLE_DEVICES"] hat nicht geklappt')
        torch.cuda.set_device(gpu_idx)

    device = torch.cuda.current_device()
    print("PyTorch verwendet GPU:", device)
    
    train_loader, val_loader, test_loader = get_sets(7)
    
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "../saved_models/CIFAR5m_subset_seed_7")
    resnet_model, resnet_results = module.train_model(
        model_name="ResNet",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        checkpoint_path=CHECKPOINT_PATH,
        model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
        seed=random_seed,
        max_epochs=180,
        verbose=False
    )
    file = open(os.path.join(f"../saved_models/CIFAR5m/resnet_results_seed_{random_seed}"), "w")
    file.write(str(resnet_results))
