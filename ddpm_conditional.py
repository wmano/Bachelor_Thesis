import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    @torch.inference_mode()
    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        #with torch.no_grad():
        with torch.inference_mode():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            try:
                predicted_noise = model(x_t, t, labels)
            except:
                import pdb; pdb.set_trace()
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

            
def safe_epoch_num(epoch):
    file = open(os.path.join("models", "DDPM_conditional", "latest_epoch.txt"), "w")
    file.write(str(epoch))

    
def get_latest_epoch():
    try:
        file = open(os.path.join("models", "DDPM_conditional", "latest_epoch.txt"), "r")
        file = file.read()
    except:
        with open(os.path.join("models", "DDPM_conditional", "latest_epoch.txt"), 'w'):
            pass
        return 0
    if file == "":
        return 0
    return int(file) # Fehler nicht abfangen um zu checken was schiefgelaufen ist
        
        
def continue_training():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.run_name = "DDPM_conditional"
    args.epochs = 700
    args.batch_size = 128
    args.image_size = 32
    args.num_classes = 10
    args.dataset_path = r"../cifar5m_jpg_data_with_classes/"
    args.device = "cuda"
    args.lr = 2e-4
    
    model_save_path = "models/DDPM_conditional/ckpt.pt"
    ema_save_path = "models/DDPM_conditional/ema_ckpt.pt"
    optim_save_path = "models/DDPM_conditional/optim.pt"
    
    latest_epoch = get_latest_epoch()
    
    set_gpu()
    
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    if latest_epoch > 0:
        print(f"load UNet Model from epoch {latest_epoch}")
        model.load_state_dict(torch.load(model_save_path))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if latest_epoch > 0:
        print(f"load optimizer from epoch {latest_epoch}")
        optimizer.load_state_dict(torch.load(optim_save_path))
    
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if latest_epoch > 0:
        print(f"load emo Model from epoch {latest_epoch}")
        ema_model.load_state_dict(torch.load(ema_save_path))

    for epoch in range(latest_epoch, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            safe_epoch_num(epoch)
            if epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))
            

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 128
    args.image_size = 32
    args.num_classes = 10
    args.dataset_path = r"../cifar5m_jpg_data_with_classes/"
    args.device = "cuda"
    args.lr = 2e-4
    train(args)

def get_free_gpu_idx():
    maximum = 0
    id_graka = None
    for i in range(4):
        print(f"GPU {i} hat {torch.cuda.mem_get_info(device=i)[0]/(1024**2)}MB frei von {torch.cuda.mem_get_info(device=i)[1]/(1024**2)}MB")
        if maximum < torch.cuda.mem_get_info(device=i)[0]:
            maximum = torch.cuda.mem_get_info(device=i)[0]
            id_graka = i
    return id_graka
   
    
def set_gpu():
    gpu_idx = get_free_gpu_idx()
    num_workers = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    if torch.cuda.current_device() != gpu_idx:
        print('os.environ["CUDA_VISIBLE_DEVICES"] hat nicht geklappt')
        torch.cuda.set_device(gpu_idx)
    print("done")
    
    device = torch.cuda.current_device()
    print("PyTorch verwendet GPU:", device)
    

def laufen():
    set_gpu()
    launch()


def create_dataset(random_seed, dataset_path, cifar_path):
    random.seed(random_seed)
    L = list(range(6))
    random.shuffle(L)
    draws_left = 60000
    for i in range(6):
        print(L[i])
        data_dict = np.load(f'{cifar_path}/cifar5m_part{L[i]}.npz', allow_pickle=True)
        if i != 5:
            ziehung = random.randint(0, draws_left)
            print('ziehung', ziehung)
        else:
            ziehung = draws_left
        von = random.randint(0, 1000448-60000)
        print('von', von)
        draws_left -= ziehung
        if i == 0:
            x = data_dict["X"][von:von+ziehung]
            y = data_dict["Y"][von:von+ziehung]
        else:
            x = np.concatenate((x, data_dict["X"][von:von+ziehung]))
            y = np.concatenate((y, data_dict["Y"][von:von+ziehung]))
        del data_dict
        
    for i in range(10):
        if not os.path.exists(f"{dataset_path}/{i}"):
            os.makedirs(f"{dataset_path}/{i}")
            
    for i in range(len(x)):
        if i % 6000 == 0:
            print(round(i*100/60000), "%")
        im = Image.fromarray(x[i])
        im.save(f"{dataset_path}/{y[i]}/{i}.png", "PNG")
    print("100%")
    del x, y
    

def training(epochs, dataset_path):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.run_name = "DDPM_conditional"
    args.epochs = epochs
    args.batch_size = 128
    args.image_size = 32
    args.num_classes = 10
    args.dataset_path = dataset_path
    args.device = "cuda"
    args.lr = 2e-4
    
    model_save_path = "models/DDPM_conditional/ckpt.pt"
    ema_save_path = "models/DDPM_conditional/ema_ckpt.pt"
    optim_save_path = "models/DDPM_conditional/optim.pt"
    
    latest_epoch = get_latest_epoch()
    
    set_gpu()
    
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    if latest_epoch > 0:
        print(f"load UNet Model from epoch {latest_epoch}")
        model.load_state_dict(torch.load(model_save_path))
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if latest_epoch > 0:
        print(f"load optimizer from epoch {latest_epoch}")
        optimizer.load_state_dict(torch.load(optim_save_path))
    
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if latest_epoch > 0:
        print(f"load emo Model from epoch {latest_epoch}")
        ema_model.load_state_dict(torch.load(ema_save_path))

    for epoch in range(latest_epoch, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            safe_epoch_num(epoch)
            if epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))
    
    
if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

