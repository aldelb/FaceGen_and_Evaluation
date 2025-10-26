import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import utils.constants as constants

def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    k = 4 if downsample else 3
    s = 2 if downsample else 1

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        return nn.Sequential(conv_block, norm_block, nn.LeakyReLU(0.2, True))
    else:
        return nn.Sequential(conv_block, nn.LeakyReLU(0.2, True))

class PoseEncoderConv(nn.Module):
    def __init__(self, dim, length):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 128),
            ConvNormRelu(128, 64),
            ConvNormRelu(64, 64, downsample=True),
            nn.Conv1d(64, 32, 3)
        )

        if length == 100:
            in_channels = 1440
        else:
            raise ValueError("Unsupported sequence length")


        self.out_net = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32),
        )

    def forward(self, poses):
        x = poses.transpose(1, 2)  # (B, D, T)
        x = self.net(x)            # (B, 32, L')
        x = x.flatten(1)           # (B, C)
        z = self.out_net(x)        # (B, 32)
        return z

class PoseDecoderConv(nn.Module):
    def __init__(self, dim, length):
        super().__init__()

        if length == 100:
            out_channels = 400
        else:
            raise ValueError("Unsupported sequence length")

        self.pre_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True),
            nn.Linear(64, out_channels),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, z):
        x = self.pre_net(z)               # (B, out_channels)
        x = x.view(z.size(0), 4, -1)      # (B, 4, L)
        x = self.net(x)                   # (B, dim, T)
        x = x.transpose(1, 2)             # (B, T, dim)
        return x

class GeneaConvAutoEncoder(pl.LightningModule):
    def __init__(self, pose_dim=28, n_frames=100, use_diff_loss=True):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = PoseEncoderConv(pose_dim, n_frames)
        self.decoder = PoseDecoderConv(pose_dim, n_frames)

        self.train_loss_history = []
        self.val_loss_history = []

    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        recon = self.decoder(z)
        return recon
    
    def forward(self, poses):
        z = self.encoder(poses)
        recon = self.decoder(z)
        return recon

    def training_step(self, batch, batch_idx):
        x = batch[1]
        z = self.encoder(x)
        recon = self.decoder(z)

        # L1 reconstruction loss
        loss_recon = F.l1_loss(recon, x, reduction='none')
        loss_recon = torch.mean(loss_recon, dim=(1, 2))  # moyenne par exemple
        total_loss = loss_recon

        # L1 sur les différences temporelles
        if self.hparams.use_diff_loss:
            x_diff = x[:, 1:] - x[:, :-1]
            recon_diff = recon[:, 1:] - recon[:, :-1]
            diff_loss = F.l1_loss(recon_diff, x_diff, reduction='none')
            diff_loss = torch.mean(diff_loss, dim=(1, 2))
            total_loss = total_loss + diff_loss

        loss = torch.sum(total_loss)  # somme sur le batch

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[1]
        z = self.encoder(x)
        recon = self.decoder(z)

        # L1 reconstruction loss
        loss_recon = F.l1_loss(recon, x, reduction='none')
        loss_recon = torch.mean(loss_recon, dim=(1, 2))
        total_loss = loss_recon

        # L1 sur les différences temporelles
        if self.hparams.use_diff_loss:
            x_diff = x[:, 1:] - x[:, :-1]
            recon_diff = recon[:, 1:] - recon[:, :-1]
            diff_loss = F.l1_loss(recon_diff, x_diff, reduction='none')
            diff_loss = torch.mean(diff_loss, dim=(1, 2))
            total_loss = total_loss + diff_loss

        loss = torch.sum(total_loss)  # somme sur le batch

        self.log("val_loss", loss)
        return loss


    def on_train_epoch_end(self):
        self.train_loss_history.append(self.trainer.logged_metrics["train_loss"].item())
        self.val_loss_history.append(self.trainer.logged_metrics["val_loss"].item())
        print(f"Epoch {self.current_epoch}: Train={self.train_loss_history[-1]:.4f}, Val={self.val_loss_history[-1]:.4f}")
        if self.current_epoch % 10 == 0:
            self.show_losses()

    def test_step(self, batch, batch_idx):
        x = batch[1]
        z = self.encoder(x)
        recon = self.decoder(z)

        loss_recon = F.l1_loss(recon, x, reduction='none')
        loss_recon = torch.mean(loss_recon, dim=(1, 2))
        total_loss = loss_recon

        if self.hparams.use_diff_loss:
            x_diff = x[:, 1:] - x[:, :-1]
            recon_diff = recon[:, 1:] - recon[:, :-1]
            diff_loss = F.l1_loss(recon_diff, x_diff, reduction='none')
            diff_loss = torch.mean(diff_loss, dim=(1, 2))
            total_loss = total_loss + diff_loss

        loss = torch.sum(total_loss)
        self.log("test_loss", loss)
        return loss

    def show_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_loss_history, label="Train", marker="o")
        plt.plot(self.val_loss_history, label="Validation", marker="o")
        plt.title("Courbes de perte")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{constants.saved_path}/losses_epoch_{self.current_epoch}.png")
        plt.close()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=constants.learning_rate, betas=(0.5, 0.999))

    def extract_latent(self, x):
        return self.encoder(x).detach()
