from utils import show_images
from config import TrainingConfig
import numpy as np
import torch
import einops
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

class Inference:
    def __init__(self, ddpm, dataloader):
        self.config = TrainingConfig()
        self.ddpm = ddpm
        self.dataloader = dataloader
        self.device = self.ddpm.device
        self.image_dims = (self.config.image_channels, self.config.image_size, self.config.image_size)

    def show_forward_pass(self, percentages=[0.25, 0.5, 0.75, 1.0]):
        for batch in self.dataloader:
            imgs = batch["images"]
            show_images(imgs)

            for percent in percentages:
                show_images(
                    self.ddpm(
                        imgs.to(self.device),
                        [int(percent * self.config.training_steps) - 1 for _ in range(len(imgs))]
                    )
                )

            break

    def generate_images(self, n_samples=16, frames_per_gif=100, gif_name="prediction.gif"):
        frame_idxs = np.linspace(0, self.config.training_steps, frames_per_gif).astype(np.uint)
        frames = []

        with torch.no_grad():
            x = torch.randn((n_samples, self.image_dims[0], self.image_dims[1], self.image_dims[2])).to(self.device)

            for idx, t in enumerate(tqdm(list(range(self.config.training_steps))[::-1], desc="Inference progress")):
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device).long()
                eta_theta = self.ddpm.backward(x, time_tensor)

                alpha_t = self.ddpm.alphas[t]
                alpha_t_bar = self.ddpm.alpha_bars[t]

                # Partially denoising image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, self.image_dims[0], self.image_dims[1], self.image_dims[2]).to(self.device)
                    beta_t = self.ddpm.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Adding more noise as in Langevin Dynamics
                    x = x + sigma_t * z

                if idx in frame_idxs or t == 0:
                    normalized = x.clone()

                    # Normalize images:
                    for i in range(len(normalized)):
                        normalized[i] -= torch.min(normalized[i])
                        normalized[i] *= 255 / torch.max(normalized[i])

                    # Arrange images into the frame:
                    frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    frame = frame.cpu().numpy().astype(np.uint8)

                    frames.append(frame)
        
        plt.imshow(frames[-1])
        plt.show()

        # Building and storing GIF:
        # with imageio.get_writer(gif_name, mode="I") as writer:
        #     for idx, frame in enumerate(frames):
        #         writer.append_data(frame)
        #         if idx == len(frames) - 1:
        #             for _ in range(frames_per_gif // 3):
        #                 writer.append_data(frames[-1])

        return x

