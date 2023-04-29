import torch
import torch.nn as nn
from config import TrainingConfig

class DDPM(nn.Module):
    def __init__(self, model, min_beta=10 ** -4, max_beta=0.02):
        super(DDPM, self).__init__()

        self.config = TrainingConfig()
        self.training_steps = self.config.training_steps
        self.model = model
        self.device = self.model.device
        self.image_dims = (self.config.image_channels, self.config.image_size, self.config.image_size)
        self.betas = torch.linspace(min_beta, max_beta, self.training_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)

    # Produce noisy image based on schedule parameters:
    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        alpha_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
        
        noisy_image = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy_image
    
    # Run backwards pass to predict added noise:
    def backward(self, x, t):
        return self.model(x, t)
