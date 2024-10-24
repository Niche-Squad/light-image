from pathlib import Path

# local imports
from data import PatachedDataModule
from model import CAE
from callbacks import ImageLoggerCallback
from trainer import NicheTrainer

paths = dict(
    train = Path.cwd() / "data" / "images" / "01-08",
    val = Path.cwd() / "data" / "images" / "01-21",
    logs = Path.cwd() / "logs",    
)


callback = ImageLoggerCallback(save_every=50, save_dir=paths["logs"])
trainer = NicheTrainer("mps")
trainer.set_model(CAE, lr=1e-5)
trainer.set_data(PatachedDataModule, 
                batch=2,
                path_train=paths["train"],
                path_val=paths["val"],)
trainer.set_out(paths["logs"])
trainer.fit(epochs=2, callbacks=[callback])


# calculate the dimensions
model = CAE()
import torch
from torch import nn

tensor = torch.randn(1, 3, 128, 128)
ec1 = model.e_conv_1(tensor)
ec1.shape