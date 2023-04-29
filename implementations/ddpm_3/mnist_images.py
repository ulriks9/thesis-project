from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

IMAGE_SIZE = (64, 64)

def to_rgb(x):
    # return x.repeat(3, 1, 1)
    np.repeat(x, 3, 0)

preprocess = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Resize(size=IMAGE_SIZE, antialias=True),
    transforms.Lambda(to_rgb),
    ])

images = datasets.MNIST(
    root="train_mnist/",
    train=True,
    download=True
    )

for i, (img, _) in tqdm(enumerate(images)):
    img.save(f"mnist_jpg/{i}.jpg")