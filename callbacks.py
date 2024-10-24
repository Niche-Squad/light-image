import torch
import torchvision
from lightning import Callback
from pathlib import Path


class ImageLoggerCallback(Callback):
    def __init__(self, save_every=50, save_dir="."):
        super().__init__()
        self.save_every = save_every
        self.save_dir = save_dir

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        # only log first image in the batch
        if batch_idx % self.save_every == 0:
            imgs, patches, _ = batch
            
            patches, dims = module.merge_patch_to_batch(patches)
            # forward pass
            with torch.no_grad():
                outputs = module(patches)
            
            # reconstruct images: (batch, channel, height, width)
            re_imgs = reconstruct_images(outputs, dims)
            
    
            comparison = torch.cat((imgs[0], re_imgs[0]), dim=2)
            comparison = comparison.unsqueeze(0)  # [1, channels, height, width * 2]
            # Normalize images to [0, 1] if needed (assuming images are in [-1, 1])
            comparison = (comparison + 1) / 2.0

            # Save or log the comparison image
            out = Path(self.save_dir) / f'comparison_{trainer.current_epoch}_{batch_idx}.png'
            torchvision.utils.save_image(comparison, out)
            # Alternatively, log the image using the logger
            # grid = torchvision.utils.make_grid(comparison)
            # self.logger.experiment.add_image(f'Comparison/{self.current_epoch}_{batch_idx}', grid, self.global_step)

            
            
def reconstruct_images(outputs, dims):
    """
    turn (batch * patchH * patchW, channel, height, width) to (batch, channel, height, width)
    
    dims, dict:
        batch: int
        channel: int
        patchH: int
        patchW: int
        height: int
        width: int
    Returns:
        torch.Tensor: Reconstructed images of shape (batch, channels, total_height, total_width)
    """
    batch_size = dims['batch']
    channels = dims['channel']
    patchH = dims['patchH']
    patchW = dims['patchW']
    height = dims['height']
    width = dims['width']
    
    # Step 1: Reshape outputs to (batch, patchH, patchW, channels, height, width)
    outputs = outputs.view(batch_size, patchH, patchW, channels, height, width)
    
    # Step 2: Permute dimensions to (batch, channels, patchH, height, patchW, width)
    outputs = outputs.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # Step 3: Reshape to (batch, channels, total_height, total_width)
    total_height = patchH * height
    total_width = patchW * width
    outputs = outputs.view(batch_size, channels, total_height, total_width)
    
    return outputs


# image_logger = ImageLoggerCallback(save_every=50)
# trainer = pl.Trainer(callbacks=[image_logger], max_epochs=10, accelerator='gpu', devices=1)