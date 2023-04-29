from data import Data
from config import TrainingConfig
from unet import UNet
from unet_seg import UNet_Seg
from ddpm import DDPM
from torchvision import transforms
from train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler
from inference import Inference

config = TrainingConfig()
device = torch.device('cuda:0')

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)

def transform(examples):
    images = [preprocess(image) for image in examples["image"]]

    return {"images": images}


data_loader = Data()
dataset = data_loader.load_data("mnist", length=1000, label=1)
dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
noise_scheduler = DDPMScheduler(num_train_timesteps=config.training_steps)

# image = next(iter(train_dataloader))["images"][0]
# image_np = image.detach().numpy()[0]

generator = torch.manual_seed(config.seed)

model = UNet_Seg(timesteps=config.training_steps).to(device)
ddpm = DDPM(model=model).to(device)

inference = Inference(ddpm=ddpm, dataloader=train_dataloader)
optimizer = torch.optim.AdamW(ddpm.parameters(), lr=config.learning_rate)
trainer = Trainer(ddpm=ddpm, dataloader=train_dataloader, optim=optimizer)

trainer.train_revised()