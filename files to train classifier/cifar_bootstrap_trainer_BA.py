import module
import cifar_image_generator

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


def train_bootstrap(model_epoch, size, device, batch, random_seed, generative_model_path, images_path):
    if not os.path.exists(f"{generative_model_path}/epoch_{model_epoch}"):
        os.makedirs(f"{generative_model_path}/epoch_{model_epoch}")

    x, y = cifar_image_generator.generate_images(model_epoch, size, device, batch, random_seed, generative_model_path, images_path)
    
    train_loader, val_loader = data_to_loaders(x, y)
    
    print("start training")
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", f"{generative_model_path}/epoch_{model_epoch}")
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
    if not os.path.exists(f"{images_path}/resnet_results_seed_{random_seed}"):
        file = open(os.path.join(f"{images_path}/resnet_results_seed_{random_seed}"), "w")
        file.write(str(resnet_results))
