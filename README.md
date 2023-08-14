# Diffusion Model
The only necessary files, to generate images are 'cifar_image_generator', 'ddpm_conditional', 'modules' and 'utils'. The model is saved in the path 'models/DDPM_conditional/'. To continue training the 'ema_ckpt_{epoch}.pt', 'ckpt_{epoch}.pt' and 'optim_{epoch}.pt' files are needed, the variable 'epoch' indicates the trained epochs of the model and has to be set in the 'latest_epoch.txt' file. All files can be found in the hessenbox cloud given in the paper.

## Train a Diffusion Model:
### Conditional Training

```python
   import ddpm_conditional
epochs = 700
dataset_path = "../cifar5m_jpg_data_with_classes/"
ddpm_conditional.training(
    epochs = epochs, 
    dataset_path = dataset_path
)
```

Hyperparameters can be configured in ```ddpm_conditional.py```

# Sampling
```python
   import cifar_image_generator as cig
   model_epoch = 50
   size = 50000
   batch = 250
   random_seed = 42
   generativ_model_path = "models/DDPM_conditional"
   images_path = "../saved_models/epoch_50"
   cig.generate_images(model_epoch, size, batch, random_seed, generativ_model_path, images_path)
```
