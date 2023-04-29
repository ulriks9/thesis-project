import torch
import matplotlib.pyplot as plt
import numpy as np

def print_progress(string):
    print(string, end="\r")

def show_images(images, fig_size=(8,8), cmap="gray"):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    
    fig = plt.figure(figsize=fig_size)
    rows = int(len(images) ** (1/2))
    cols = round(len(images) / rows)

    idx = 0
    for r in range(rows):
        fig.add_subplot(rows, cols, idx + 1)

        if idx < len(images):
            plt.imshow(images[idx][0], cmap=cmap)
            idx += 1
    
    plt.show()
