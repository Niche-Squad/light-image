import torch
import torch.nn as nn
import lightning as l


class CAE(l.LightningModule):
    """
    mostly based on the repository alexandru-dinu/cae:
        https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_32x32x32_zero_pad_bin.py
    """

    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=3, out_channels=64, 
                kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=64, out_channels=128, 
                kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
            ),
            nn.Tanh(),
        )

        # DECODER

        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=128, 
                kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=256, 
                kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=16, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=16, out_channels=3, 
                kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        # ENCODER
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from Tanh activation

        # Stochastic binarization
        with torch.no_grad():
            rand = torch.rand_like(ec3)
            prob = (1 + ec3) / 2  # Scale to [0, 1]
            eps = torch.zeros_like(ec3)
            mask = rand <= prob
            eps[mask] = (1 - ec3)[mask]
            eps[~mask] = (-ec3 - 1)[~mask]

        # Encoded tensor: binarized representation
        self.encoded = 0.5 * (ec3 + eps + 1)  # Convert from [-1, 1] to [0, 1]

        # DECODER
        decoded = self.decode(self.encoded)
        return decoded

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # Convert from [0, 1] back to [-1, 1]
        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec

    def training_step(self, batch, batch_idx):
        """
        the batch is in the dimension of:
        (batch, channel, 6, 10, width, height)
        """
        img, patches, _ = batch
        patches, dims = self.merge_patch_to_batch(patches)
        # forward pass
        x_hat = self(patches)
        loss = nn.MSELoss()(x_hat, patches)
        self.log('train_loss', loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        img, patches, _ = batch
        patches, dims = self.merge_patch_to_batch(patches)
        # forward pass
        x_hat = self(patches)
        loss = nn.MSELoss()(x_hat, patches)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        img, patches, _ = batch
        patches, dims = self.merge_patch_to_batch(patches)
        # forward pass
        x_hat = self(patches)
        loss = nn.MSELoss()(x_hat, patches)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                }
        }

    def merge_patch_to_batch(self, patches):
        """
        converting the patches dimension 
        from (batch, channel, 6, 10, width, height)
        to (batch, channel, width, height)
        """
        dims = dict(
                batch=patches.shape[0],
                channel=patches.shape[1],
                patchH=patches.shape[2],
                patchW=patches.shape[3],
                height=patches.shape[4],
                width=patches.shape[5],
        )
        # reshape patches
        patches = patches.permute(0, 2, 3, 1, 4, 5) # (batch, 6, 10, channel, height, width)
        patches = patches.contiguous()
        patches = patches.view(-1, dims['channel'], dims['height'], dims['width'])
        return patches, dims
