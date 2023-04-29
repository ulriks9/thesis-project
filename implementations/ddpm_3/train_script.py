import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import wandb

DEVICE = 'cuda:0'
EPOCHS = 50
LABEL = 1
TIMESTEPS = 1000
INITIAL_DIM = 64
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 50

wandb.login()

run = wandb.init(
    project="conifdent-diffusion",
    config={
        "Epochs": EPOCHS,
        "Timesteps": TIMESTEPS,
        "Initial Conv Dim": INITIAL_DIM,
        "Image Size": IMAGE_SIZE,
        "Batch Size": BATCH_SIZE
    }
)

def to_rgb(x):
    return x.repeat(3, 1, 1)

def main():
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE, antialias=True),
        transforms.Lambda(to_rgb),
        transforms.Normalize((0, 0, 0), (1,1,1))
        ])
    
    images = datasets.MNIST(
        root="train_mnist/",
        train=True,
        download=True,
        transform=preprocess
        )
    
    idx = images.targets == LABEL
    images.targets = images.targets[idx]
    images.data = images.data[idx]

    dl = torch.utils.data.DataLoader(
        images,
        batch_size=BATCH_SIZE,
        shuffle=False
        )
    
    example_image = next(iter(dl))[0][0].numpy()
    plt.imshow(example_image.transpose((1, 2, 0)))
    plt.show()
    
    model = Unet(
        dim=INITIAL_DIM,
        dim_mults = (1, 2, 4, 8)
        ).to(DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE[0],
        timesteps=TIMESTEPS,
        loss_type='l1'
        ).to(DEVICE)

    for epoch in tqdm(range(EPOCHS), desc='Training Progress: '):
        pbar = tqdm(dl, leave=False, desc=f"Epoch {epoch + 1}/{EPOCHS}", colour="#005500")
        for batch in pbar:
            batch = batch[0].to(DEVICE)
            loss = diffusion(batch)
            loss.backward()
            logs = {"loss": loss.detach().item()}
            pbar.set_postfix(**logs)
            wandb.log({"loss": loss.detach().item()})

    # Make predictions:
    sampled_images = diffusion.sample(batch_size=1)
    for image in sampled_images:
        image = image.cpu().detach().numpy()
        image = image.transpose((1,2,0))

        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    main()