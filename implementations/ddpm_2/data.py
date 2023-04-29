from datasets import load_dataset
from scipy.io import wavfile
from config import TrainingConfig

import matplotlib.pyplot as plt
import os

class Data():
    def __init__(self):
        self.config = TrainingConfig()

    def generate_spectrograms(self, dataset, verbose=True):
        if dataset == "beethoven":
            # Create directories
            if not os.path.exists("data/beethoven/spectrograms"):
                if verbose:
                    print("INFO: Directory for spectrograms created.")
                os.makedirs("data/beethoven/spectrograms")

            data = load_dataset("krandiash/beethoven", split="train")

            if verbose:
                 print("INFO: Dataset loaded from Huggingface.")

            print(data[0]["audio"]["path"])
            data_length = len(data)
            
            if verbose:
                 print("INFO: Starting spectrogram conversion.")

            for i in range(data_length):
                sample_rate, samples = wavfile.read(data[i]["audio"]["path"])
                _, _, _, img = plt.specgram(samples, sample_rate)
                plt.savefig("data/beethoven/spectrograms/{}.png".format(i))

            if verbose:
                 print("INFO: Spectrograms successfully generated.")

    # Returns specified dataset
    def load_data(self, name, length=None, label=None):
        if name == "beethoven":
            self.data = load_dataset("krandiash/beethoven", split="train")
        elif name == "mnist":
            self.data = load_dataset("mnist", split="train")

        if label is not None:
            self.data = self.data.filter(lambda example: example["label"] == 1)

        self.data.shuffle()

        if length != None:
            self.data = self.data.select(range(length))

        return self.data
