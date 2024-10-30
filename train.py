from pathlib import Path
import argparse
import os
from dotenv import load_dotenv

# local imports
from data import PatachedDataModule
from model import CAE_16, CAE_32
from callbacks import ImageLoggerCallback
from trainer import NicheTrainer

def main(args):
    ls_jobs = []
    for lr in [1e-3, 1e-4, 1e-5]:
        for modelclass in [CAE_16, CAE_32]:
            ls_jobs.append((lr, modelclass))
    i_job = int(args.job)
    lr, modelclass = ls_jobs[i_job]
                
    load_dotenv(".env")
    DIR_DATA = Path(os.getenv("DIR_DATA"))
    paths = dict(
        train = DIR_DATA / "01-08",
        val = DIR_DATA / "01-21",
        logs = Path.cwd() / "logs" / f"{modelclass.__name__}_{lr}" 
    )

    callback = ImageLoggerCallback(save_every=500, 
                                   save_dir=paths["logs"])
    trainer = NicheTrainer("cuda")
    trainer.set_model(modelclass, lr=lr)
    trainer.set_data(PatachedDataModule, 
                    batch=2,
                    path_train=paths["train"],
                    path_val=paths["val"],)
    trainer.set_out(paths["logs"])
    trainer.fit(epochs=50, callbacks=[callback])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--lr", type=float)
    # parser.add_argument("--model", type=str, default="CAE_32")  
    args = parser.parse_args()  
    main(args)