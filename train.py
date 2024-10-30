from pathlib import Path

# local imports
from data import PatachedDataModule
from model import CAE_16, CAE_32
from callbacks import ImageLoggerCallback
from trainer import NicheTrainer
import os
from dotenv import load_dotenv

def main():
    load_dotenv(".env")
    DIR_DATA = Path(os.getenv("DIR_DATA"))
    paths = dict(
        train = DIR_DATA / "01-08",
        val = DIR_DATA / "01-21",
        logs = Path.cwd() / "logs" / "cae16"   
    )

    callback = ImageLoggerCallback(save_every=50, 
                                   save_dir=paths["logs"])
    trainer = NicheTrainer("mps")
    trainer.set_model(CAE_16, lr=1e-5)
    trainer.set_data(PatachedDataModule, 
                    batch=2,
                    path_train=paths["train"],
                    path_val=paths["val"],)
    trainer.set_out(paths["logs"])
    trainer.fit(epochs=2, callbacks=[callback])

if __name__ == "__main__":
    main()