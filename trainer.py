# native imports
import os

# torch imports
import torch
import lightning as l
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, checkpoint
from lightning.pytorch.loggers import TensorBoardLogger

class NicheTrainer:
    def __init__(self, device="cpu"):
        # core
        self.device = device  # cpu, cuda, or mps
        self.model = None  # lightning module instance
        self.data = None # lightning data module
        self.loggers = None  # lightning loggers
        self.callbacks = None # lightning callbacks
        # output
        self.out = dict({
            "dir": None,
            "best_loss": None,
            "best_path": None,
        })
        self.batch = None
        
    def set_model(
            self, 
            modelclass: l.LightningModule,
            checkpoint: str = None,
            **kwargs,):
        """
        parameters
        ---
        model_class: l.LightningModule
            the lightning module class
        checkpoint: str
            local path to the checkpoint, e.g., model.ckpt
        """
        if checkpoint:
            self.model = modelclass.load_from_checkpoint(checkpoint, **kwargs)
            print(f"model loaded from {checkpoint}")
        else:
            self.model = modelclass(**kwargs)
        self.model.to(self.device)
    
    def set_data(
            self,
            dataclass: l.LightningDataModule,
            batch: int = 32,
            **kwargs,):
        """
        parameters
        ---
        dataclass: l.LightningDataModule
            the lightning data module class, e.g., transformers.DetrData
        """
        self.data = dataclass(batch=batch, **kwargs)
    
    def set_out(
            self,
            dir_out: str,):
        self.out["dir"] = dir_out
        os.makedirs(self.out["dir"], exist_ok=True)
    
    def fit(
            self,
            epochs: int = 10,
            rm_threshold: float = 1e10, # default not to remove
            callbacks: list = None,
            **kwargs,):
        """
        parameters
        ---
        epochs: int
            number of epochs to train
        """
        ls_callbacks = [self.checkpoint()]
        if callbacks:
            if not isinstance(callbacks, list):
                raise ValueError("callbacks must be a list")
            ls_callbacks += callbacks

        self.trainer = Trainer(
            max_epochs=epochs,
            callbacks=ls_callbacks,
            logger=self.logger(),
            **kwargs,
        )
        self.trainer.fit(self.model, self.data)
        self.process_best(ls_callbacks[0], rm_threshold)
        self.set_model(self.model.__class__, self.out["best_path"])
        
    def process_best(
            self, 
            checkpoint_callback: ModelCheckpoint,
            rm_threshold: float = 1e10,):
        """
        store the best loss/path and rm models with loss > rm_threshold
        """
        self.out["best_loss"] = checkpoint_callback.best_model_score
        self.out["best_path"] = checkpoint_callback.best_model_path
        if self.out["best_loss"] > rm_threshold:
            os.remove(self.out["best_path"])
    
    def checkpoint(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.out["dir"],
            mode="min",
            save_top_k=1,
            verbose=False,
            save_last=False,
            filename="model-{val_loss:.4f}",
        )
        return checkpoint_callback

    def logger(self):
        logger = TensorBoardLogger(
            save_dir=self.out["dir"],
            name=".",
            version=".",
            log_graph=True,
            default_hp_metric=False,
        )
        return logger