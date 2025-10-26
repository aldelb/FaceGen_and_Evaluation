import numpy as np
import pandas as pd
from os.path import join
import utils.constants.constants as constants
import matplotlib
from matplotlib import pyplot as plt

def plotHistEpoch(file):
    file_path = join(constants.saved_path, file)
    metrics_df = pd.read_csv(file_path, delimiter=";")[1:] #we skip the first raw, epoch 0, to avoid big values in the plot
    epoch = metrics_df["epoch"].values
    g_loss = metrics_df["g_loss"].values
    val_g_loss = metrics_df["val_g_loss"].values
    d_loss = metrics_df["d_loss"].values
    loss_eye = metrics_df["loss_eye"].values
    val_loss_eye = metrics_df["val_loss_eye"].values
    loss_pose_r = metrics_df["loss_pose_r"].values
    val_loss_pose_r = metrics_df["val_loss_pose_r"].values
    loss_au = metrics_df["loss_au"].values
    val_loss_au = metrics_df["val_loss_au"].values
    real_pred = metrics_df["real_pred"].values
    fake_pred = metrics_df["fake_pred"].values
    basic_fake_pred = metrics_df["basic_fake_pred"].values #only generated during the generator training, without designed examples

    plotHistPredEpochGAN(epoch, real_pred, fake_pred, basic_fake_pred)
    plotHistLossEpoch(epoch, g_loss, val_g_loss, d_loss)
    plotHistAllLossEpoch(epoch, loss_eye, val_loss_eye, loss_pose_r, val_loss_pose_r, loss_au, val_loss_au)
    

def plotHistPredEpochGAN(epoch, real_pred, fake_pred, basic_fake_pred):
    plt.figure(dpi=100)

    plt.plot(epoch, real_pred, color="blue", label='Real')
    plt.plot(epoch, fake_pred, color="red", label='Fake')
    plt.plot(epoch, basic_fake_pred, color="lightpink", label='Basic fake (only generated)')

    #plt.yticks(np.arange(0, 1, step=0.2)) 
    plt.xlabel("Epoch")
    plt.ylabel("Critic prediction")
    plt.legend()
    plt.savefig(constants.saved_path+f'pred_gan.png')
    plt.close()

def plotHistLossEpoch(epoch, g_loss, val_g_loss, d_loss):
    plt.figure(dpi=100)
    plt.plot(epoch, g_loss, label='Generator loss', color="blue")
    plt.plot(epoch, val_g_loss, label='Val Gen loss', color="lightblue")
    plt.plot(epoch, d_loss, label='Critic loss', color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'loss_gan.png')
    plt.close()

def plotHistAllLossEpoch(epoch, loss_eye, val_loss_eye, loss_pose_r, val_loss_pose_r, loss_au, val_loss_au):
    plt.figure(dpi=100)
    plt.plot(epoch, loss_eye, color="darkgreen", label='Loss eye')
    plt.plot(epoch, val_loss_eye, color="limegreen", label='Val loss eye')

    plt.plot(epoch, loss_pose_r, color="darkblue", label='Loss pose_r')
    plt.plot(epoch, val_loss_pose_r, color="cornflowerblue", label='Val loss pose_r')

    plt.plot(epoch, loss_au, color="red", label='Loss au')
    plt.plot(epoch, val_loss_au, color="lightcoral", label='Val loss au')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(constants.saved_path+f'loss_mse.png')
    plt.close()