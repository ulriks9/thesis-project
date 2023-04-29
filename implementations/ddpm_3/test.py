import torchaudio

dataset = torchaudio.datasets.GTZAN(
    root="train_gtzan/",
    download=True
    )