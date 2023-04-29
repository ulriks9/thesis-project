from config import TrainingConfig
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import show_images
from inference import Inference

class Trainer:
    def __init__(self, ddpm, dataloader, optim, checkpoint_path="checkpoints/"):
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.checkpoint_path = checkpoint_path + "/model"
        self.ddpm = ddpm
        self.device = ddpm.device
        self.config = TrainingConfig()
        self.optimizer = optim
        self.dataloader = dataloader
        self.inference = Inference(self.ddpm, self.dataloader)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.dataloader) * self.config.num_epochs)
        )
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps, 
            log_with="tensorboard",
            logging_dir=os.path.join(self.config.output_dir, "logs")
        )

    def train_revised(self, display=False):
        mse = nn.MSELoss()
        best_loss = float("inf")

        for epoch in tqdm(range(self.config.num_epochs), desc="Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, leave=False, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", colour="#005500")

            for step, batch in enumerate(pbar):
                self.ddpm.train()
                batch = batch["images"]
                x0 = batch.to(self.device)
                batch_size = len(x0)

                eta = torch.randn_like(x0).to(self.device)
                t = torch.randint(0, self.config.training_steps, (batch_size,),).to(self.device)

                noisy_imgs = self.ddpm(x0, t, eta)

                eta_theta = self.ddpm.backward(noisy_imgs, t.reshape(batch_size, -1))

                loss = mse(eta_theta, eta)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(x0) / len(self.dataloader.dataset)

                logs = {"loss": loss.detach().item()}
                pbar.set_postfix(**logs)

            if display:
                show_images(self.inference.generate_images())

            log_string = f"INFO: Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(self.ddpm.state_dict(), self.checkpoint_path + f"model{epoch + 1}.pt")
                log_string += "INFO: Best model stored."

            print(log_string)

    # def train(self):
    #     if self.accelerator.is_main_process:
    #         if self.config.output_dir is not None:
    #             os.makedirs(self.config.output_dir, exist_ok=True)

    #         self.accelerator.init_trackers("train_example")

    #     model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
    #         self.model, self.optimizer, self.dataloader, self.lr_scheduler
    #     )
        
    #     global_step = 0

    #     for epoch in range(self.config.num_epochs):
    #         progress_bar = tqdm(total=len(train_dataloader), disable=not self.accelerator.is_local_main_process, position=1, leave=True)
    #         progress_bar.set_description(f"Epoch {epoch}")

    #         for _, batch in enumerate(train_dataloader):
    #             clean_images = batch["images"]

    #             noise = torch.randn(clean_images.shape).to(clean_images.device)
    #             bs = clean_images.shape[0]

    #             # Sample a random timestep for each image
    #             timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

    #             # Add noise to the clean images according to the noise magnitude at each timestep
    #             # (this is the forward diffusion process)
    #             noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
                
    #             with self.accelerator.accumulate(model):
    #                 # Predict the noise residual
    #                 noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
    #                 loss = F.mse_loss(noise_pred, noise)
    #                 self.accelerator.backward(loss)

    #                 self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
    #                 optimizer.step()
    #                 lr_scheduler.step()
    #                 optimizer.zero_grad()
                
    #             progress_bar.update(1)
    #             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
    #             progress_bar.set_postfix(**logs)
    #             self.accelerator.log(logs, step=global_step)
    #             global_step += 1

    #         # After each epoch you optionally sample some demo images with evaluate() and save the model
    #         if self.accelerator.is_main_process:
    #             if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
    #                 torch.save({
    #                     "epoch": epoch,
    #                     "model_state_dict": self.model.state_dict(),
    #                     "optimizer_state_dict": self.optimizer.state_dict(),
    #                     "loss": loss
    #                 }, self.checkpoint_path + str(epoch + 1) + ".pt")

    #             if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
    #                 image = next(iter(self.dataloader))["images"][0]
    #                 image_np = image.detach().numpy()[0]

    #                 self.model.eval()
    #                 prediction = model.predict(shape=image_np.shape)
    #                 print(f"INFO: Shape of prediction {prediction.shape}")
                    
