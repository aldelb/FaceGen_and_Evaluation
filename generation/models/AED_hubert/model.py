from os.path import join
import csv
from datetime import datetime
from math import floor
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import random

import utils.constants.constants as constants
from utils.data.data_utils import format_data, reshape_output, concat_with_labels
from utils.data.noise_generator import NoiseGenerator
from utils.params_utils import save_params
from utils.evaluation.plot_utils import plotHistEpoch
from utils.model.model_parts import Down, OutConv, Up, Conv, ConvLayerNorm, DownLayerNorm
from torch.nn.utils import spectral_norm



class AudioEmbeddingGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.in1_audio = Conv(constants.audio_dim, 256, constants.kernel_size)
        self.in2_audio = Conv(256, 128, constants.kernel_size)
        self.down1_audio = Down(128, 64, constants.kernel_size) 
        self.down2_audio = Down(64, 128, constants.kernel_size)
        self.down3_audio = Down(128, 256, constants.kernel_size)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x = self.in1_audio(x)
        x = self.in2_audio(x)
        x1 = self.down1_audio(x)
        x2 = self.down2_audio(x1)
        x3 = self.down3_audio(x2)
        return x1, x2, x3

class Generator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        bilinear = True
        factor = 2 if bilinear else 1
        self.noise_g = NoiseGenerator()

        ##encode audio
        self.audio_embedding = AudioEmbeddingGenerator()

        ###concat with noise and labels here
        self.down4 = Down(256 + constants.number_of_dim_labels, 512, constants.kernel_size) # 256 for the audio embedding 
        self.down5 = Down(512, 1024, constants.kernel_size)
        self.down6 = Down(1024, 2048 // factor, constants.kernel_size)


        ##Decoder eye
        self.up1_eye = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_eye = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_eye = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_eye = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_eye = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_eye = OutConv(64, constants.eye_size, constants.kernel_size)

        ##Decoder pose_r
        self.up1_pose_r = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_pose_r = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_pose_r = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_pose_r = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_pose_r = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_pose_r = OutConv(64, constants.pose_r_size, constants.kernel_size)

        ##Decoder AUs
        self.up1_au = Up(2048, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_au = Up(1024, 512 // factor, constants.kernel_size, bilinear)
        self.up3_au = Up(512, 256 // factor, constants.kernel_size, bilinear)
        self.up4_au = Up(256, 128 // factor, constants.kernel_size, bilinear)
        self.up5_au = Up(128, 64, constants.kernel_size, bilinear)
        self.outc_au = OutConv(64, constants.au_size, constants.kernel_size)

    def forward(self, x_audio, labels):
        x1, x2, x3 = self.audio_embedding(x_audio)

        ###concat with noise and labels here
        noise = self.noise_g.getNoise(x3, std=constants.std_noise, interval=[-1,1])
        x_audio_noise = torch.add(x3, noise)

        x_audio_noise = concat_with_labels(x_audio_noise, x_audio_noise.shape[2], labels, c_dim=1, l_dim=2)

        #Encoder (audio + noise part)
        x4 = self.down4(x_audio_noise)
        x5 = self.down5(x4)
        latent_representation = self.down6(x5)

        #Decoder gaze
        x = self.up1_eye(latent_representation, x5)
        x = self.up2_eye(x, x4)
        x = self.up3_eye(x, x3)
        x = self.up4_eye(x, x2)
        x = self.up5_eye(x, x1)
        logits_eye = self.outc_eye(x)
        logits_eye = torch.tanh(logits_eye)

        #Decoder pose_r
        x = self.up1_pose_r(latent_representation, x5)
        x = self.up2_pose_r(x, x4)
        x = self.up3_pose_r(x, x3)
        x = self.up4_pose_r(x, x2)
        x = self.up5_pose_r(x, x1)
        logits_pose_r = self.outc_pose_r(x)
        logits_pose_r = torch.tanh(logits_pose_r)

        #Decoder AUs
        x = self.up1_au(latent_representation, x5)
        x = self.up2_au(x, x4)
        x = self.up3_au(x, x3)
        x = self.up4_au(x, x2)
        x = self.up5_au(x, x1)
        logits_au = self.outc_au(x)
        logits_au = torch.tanh(logits_au)
        
        logits_eye = torch.swapaxes(logits_eye, 1, 2)
        logits_pose_r = torch.swapaxes(logits_pose_r, 1, 2)
        logits_au = torch.swapaxes(logits_au, 1, 2)
        return latent_representation, logits_eye, logits_pose_r, logits_au


class AudioEmbeddingDiscriminator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1_audio = ConvLayerNorm(constants.audio_dim, 512, constants.kernel_size, constants.seq_len*2)
        self.conv2_audio = ConvLayerNorm(512, 128, constants.kernel_size, constants.seq_len)
        self.conv3_audio = ConvLayerNorm(128, 64, constants.kernel_size, constants.seq_len//2)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x = self.conv1_audio(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv2_audio(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv3_audio(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        return x

class BehavEmbeddingDiscriminator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.conv1_behaviour = ConvLayerNorm(constants.pose_size + constants.au_size, 32, constants.kernel_size, constants.seq_len)
        self.conv2_behaviour = ConvLayerNorm(32, 64, constants.kernel_size, constants.seq_len//2)
        
    def forward(self, x):
        x = self.conv1_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv2_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        return x
    
class Discriminator(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.embed_audio = AudioEmbeddingDiscriminator()
        self.embed_behaviour = BehavEmbeddingDiscriminator()

        self.conv_concat = ConvLayerNorm(128 + constants.number_of_dim_labels, 64, constants.kernel_size, constants.seq_len//4)
        # self.fc1 = torch.nn.Linear(64 * floor(constants.seq_len//4), 64)
        # self.fc2 = torch.nn.Linear(64, 1)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(64 * floor(constants.seq_len//4), 64))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(64, 1))

    def forward(self, x_pose, c_audio, labels):
        c = self.embed_audio(c_audio)

        ###concat with labels here
        x_audio_labels = concat_with_labels(c, c.shape[2], labels, c_dim=1, l_dim=2)

        x = torch.swapaxes(x_pose, 1, 2)
        x = self.embed_behaviour(x)

        x = torch.cat([x, x_audio_labels], dim=1)
        x = self.conv_concat(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x) #not adapted to the WGAN
        return x
    
class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.c_lambda = constants.gp_weight
        self.criterion = nn.MSELoss()
        self.create_loss()

        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.generator.apply(self.weights_init_xavier)
        self.discriminator.apply(self.weights_init_xavier)

        save_params(constants.saved_path, self.generator, self.discriminator)

    def forward(self, x_audio, labels):
        return self.generator(x_audio, labels)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=constants.g_lr, betas=(constants.b1_g, 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=constants.d_lr, betas=(constants.b1_d, 0.9))

        # Schedulers
        # g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=1.0)  # no decay for G
        # d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.98)  # decay progressively for D

        return g_opt, d_opt
    
    def weights_init_xavier(self, m):
        # Initialize weights using Xavier initialization
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def log_gradients(self, model, name="generator"):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log(f"{name}_grad_norm", total_norm)

    def on_train_epoch_start(self):
        #torch.cuda.empty_cache()
        #print(str(self.device)+ " memory allocated "+ str(torch.cuda.memory_allocated())+ " max memory allocated "+ str(torch.cuda.max_memory_allocated()))
        self.start_epoch = datetime.now()

    def temporal_smoothness_loss(self, current_behav, generated_current_behav):
        # Calculate the difference between each frame (discrete derivative)
        real_diff = current_behav[:, 1:, :] - current_behav[:, :-1, :]
        gen_diff = generated_current_behav[:, 1:, :] - generated_current_behav[:, :-1, :]
        diff_loss = F.l1_loss(real_diff, gen_diff, reduction='none')
        diff_loss = torch.mean(diff_loss)
        return diff_loss
    
    def mse_loss(self, gen, target):
        # Compute the MSE loss
        return self.criterion(gen, target).mean()
            
    def adversarial_loss(self, fake_pred):
        # Compute the generator loss using the discriminator's predictions
        # The generator tries to maximize the discriminator's predictions for fake samples
        return -1. * torch.mean(fake_pred)

    
    def generator_step(self, inputs_audio, targets, labels, val=False):
            inputs_audio, targets, target_eye, target_pose_r, target_au = format_data(inputs_audio, targets)
    
            latent_representation, gen_eye, gen_pose_r, gen_au = self.generator(inputs_audio, labels)
            fake_targets = torch.cat((gen_eye, gen_pose_r, gen_au), 2)
            
            
            d_fake_pred = self.discriminator(fake_targets, inputs_audio, labels).squeeze()
            with torch.no_grad():
                d_real_pred = self.discriminator(targets, inputs_audio, labels).squeeze()

            g_iteration_loss = self.adversarial_loss(d_fake_pred)
            loss_eye = self.mse_loss(gen_eye, target_eye)
            loss_pose_r = self.mse_loss(gen_pose_r, target_pose_r)
            loss_au = self.mse_loss(gen_au, target_au)
            loss_temporal_smoothness = self.temporal_smoothness_loss(targets, fake_targets)

            # Compute the generator loss

            g_loss = constants.adversarial_coeff * g_iteration_loss + constants.eye_coeff * loss_eye + constants.pose_coeff * loss_pose_r + constants.au_coeff * loss_au + constants.temporal_smoothness * loss_temporal_smoothness

            # Log the losses
            if val:
                prefix = "val_"
            else:
                prefix = ""
            self.log_dict[prefix+"g_loss"].append(g_loss)
            self.log_dict[prefix+"g_iteration_loss"].append(constants.adversarial_coeff * g_iteration_loss)
            self.log_dict[prefix+"loss_eye"].append(constants.eye_coeff * loss_eye)
            self.log_dict[prefix+"loss_pose_r"].append(constants.pose_coeff * loss_pose_r)
            self.log_dict[prefix+"loss_au"].append(constants.au_coeff * loss_au)
            self.log_dict[prefix+"temporal_smoothness"].append(constants.temporal_smoothness * loss_temporal_smoothness)
            self.log_dict[prefix+"basic_real_pred"].append(torch.mean(d_real_pred))
            self.log_dict[prefix+"basic_fake_pred"].append(torch.mean(d_fake_pred))

            return latent_representation, g_loss
    

    def discriminator_step(self, inputs_audio, targets, labels, designed_batch):
            inputs_audio = inputs_audio.squeeze(1)
            
            with torch.no_grad():
                _, output_eye, output_pose_r, output_au = self.generator(inputs_audio, labels)
            fake_targets = torch.cat((output_eye, output_pose_r, output_au), 2)
            
            selected_audio_inputs, selected_labels, selected_real_targets, selected_fake_targets = self.create_fake_targets(inputs_audio, targets, fake_targets, labels, designed_batch) 

            #fake predictions 
            fake_pred = self.discriminator(selected_fake_targets, selected_audio_inputs, selected_labels).squeeze()
            #real predictions 
            real_pred = self.discriminator(selected_real_targets, selected_audio_inputs, selected_labels).squeeze()

            gp = self.compute_gradient_penalty(targets, fake_targets, inputs_audio, labels)
            d_loss = torch.mean(fake_pred) - torch.mean(real_pred) +  self.c_lambda * gp

            # Log the losses
            self.log_dict["d_loss"].append(d_loss)
            self.log_dict["real_pred"].append(torch.mean(real_pred))
            self.log_dict["fake_pred"].append(torch.mean(fake_pred))
            self.log_dict["wasserstein_distance"].append(torch.mean(real_pred) - torch.mean(fake_pred))
            self.log_dict["penalty"].append(self.c_lambda * gp)
            
            return d_loss
    
    def create_fake_targets(self, audio, behav, generated_behav, labels, designed_batch): 
        # Create the fake targets for the discriminator
        # The designed batch is a batch of generated examples with the some "mixed" real examples

        designed_criterias, designed_values = designed_batch
        d_audio, d_real_behaviour, d_fake_behaviour, d_labels = designed_values
        d_audio = d_audio.squeeze(1)
        batch_size = audio.size(0)
        # At least a quarter of the batch size correspond to the generated examples
        nb_generated = int(batch_size*constants.pourcent_generated)
        # The rest is designed with real examples shuffle depending on chosen criteria, same number per example type
        nb_designed = int((batch_size - nb_generated)/len(constants.designed_targets))*len(constants.designed_targets)
        nb_generated = batch_size - nb_designed
        
        # select the fake examples generated
        new_order = torch.randperm(batch_size) # shuffle the order of the elements
        idx_generated = random.sample(range(batch_size), nb_generated) #select the indexes of the generated examples randomly
        idx_designed = random.sample(range(batch_size), nb_designed) #select the indexes of the designed examples randomly
        
        criteria = ["generated"]*nb_generated
        criteria.extend([designed_criterias[i] for i in idx_designed])
        criteria = [criteria[i] for i in new_order]
        
        final_audio = torch.cat((audio[idx_generated], d_audio[idx_designed]), dim=0)[new_order]
        final_real_behaviour = torch.cat((behav[idx_generated], d_real_behaviour[idx_designed]), dim=0)[new_order]
        final_fake_behaviour = torch.cat((generated_behav[idx_generated], d_fake_behaviour[idx_designed]), dim=0)[new_order]
        final_labels = {}
        for current_lab in constants.list_of_labels:
            final_labels[current_lab] = torch.cat((labels[current_lab][idx_generated], d_labels[current_lab][idx_designed]), dim=0)[new_order]

        return final_audio, final_labels, final_real_behaviour, final_fake_behaviour
    

    def compute_gradient_penalty(self, real_samples, fake_samples, inputs_audio, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(len(real_samples), 1, 1, requires_grad=True).to(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(real_samples)
        d_interpolates = self.discriminator(interpolates, inputs_audio, labels) #real inputs so i give real labels

        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates), 
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]
        
        gradient = gradient.reshape(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        self.log("grad_norm", gradient_norm.mean())  # log the gradient norm
        penalty = torch.mean((gradient_norm - 1)**2).mean()
        return penalty


    def training_step(self, batch, batch_idx):
        # Training step
        # This function is called during training to compute the training loss
        if(len(constants.designed_targets) > 0):
            real_batch, designed_batch = batch["real"], batch["designed"]
        else:
            real_batch = batch["real"]
            designed_batch = None

        inputs_audio, targets = real_batch[0], real_batch[1]
        labels = {}
        for i, label_type in enumerate(constants.list_of_labels):
            labels[label_type] = batch[2+i]

        g_opt, d_opt = self.optimizers()

        for i in range(constants.n_critics):
            d_loss = self.discriminator_step(inputs_audio, targets, labels, designed_batch[i])

            d_opt.zero_grad()
            self.manual_backward(d_loss)
            self.log_gradients(self.discriminator, "discriminator")
            d_opt.step()

        _, g_loss = self.generator_step(inputs_audio, targets, labels)
        
        self.log("lr_d", d_opt.param_groups[0]["lr"])
        self.log("lr_g", g_opt.param_groups[0]["lr"])
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.log_gradients(self.generator, "generator")
        g_opt.step()
            


    def on_train_epoch_end(self):
        @pl.utilities.rank_zero_only
        def _save_losses(dict_loss):
            # Save the losses to a CSV file
            file = "lossEpoch.csv"
            metrics = {"device": str(self.device),
                        "memory allocated": str(torch.cuda.memory_allocated()),
                        "epoch": self.current_epoch,
                        "duration": (datetime.now() - self.start_epoch).total_seconds()}
            for key in dict_loss.keys():
                new_key = key
                if key.startswith("avg_"):
                    new_key = key[4:]
                metrics[new_key] = dict_loss[key].item()
        
            self.log_metrics_to_csv(metrics, file, mode=current_mode)

            if self.current_epoch % constants.log_interval == 0 and self.current_epoch != 0:
                # Save the model every log_interval epochs
                plotHistEpoch(file)

        if(self.current_epoch == 0):
            current_mode = "w"
        else:
            current_mode = "a"

        dict_loss = {}
        for key in self.log_dict.keys():
            dict_loss["avg_"+key] = torch.mean(torch.tensor(self.log_dict[key]))

        _save_losses(dict_loss)
        self.clear_loss()


    def validation_step(self, batch, batch_idx):
        # Validation step
        # This function is called during validation to compute the validation loss
        inputs_audio, targets = batch[0], batch[1]
        labels = {}
        for i, label_type in enumerate(constants.list_of_labels):
            labels[label_type] = batch[2+i]

        with torch.no_grad():
            _, val_g_loss = self.generator_step(inputs_audio, targets, labels, val=True)

        return val_g_loss


    def predict_step(self, batch, batch_idx):
        # Prediction step
        # This function is called when you want to generate predictions using the trained model
        inputs_audio, details_time, key = batch[0], batch[1], batch[2]
        labels = {}
        for i, label_type in enumerate(constants.main_list_for_loading): # Permit to visualize labels even if they are not used in the model
            labels[label_type] = batch[3+i]

        with torch.no_grad():
            latent_representation, output_eye, output_pose_r, output_au = self(inputs_audio.squeeze(1), labels)
        pred = reshape_output(output_eye, output_pose_r, output_au, self.pose_scaler)

        return key, pred, details_time, latent_representation, labels



    ################ Log/Loss processing #########################
    def create_loss(self):
        #create the loss dict for the training
        self.log_dict = {loss_name: [] for loss_name in ["g_loss", "val_g_loss", "g_iteration_loss", "val_g_iteration_loss", "d_loss", "penalty", "wasserstein_distance", "temporal_smoothness", "val_temporal_smoothness"]}
        self.log_dict.update({loss_name: [] for loss_name in ["fake_pred", "real_pred", "basic_real_pred", "basic_fake_pred", "val_basic_real_pred", "val_basic_fake_pred"]})
        self.log_dict.update({loss_name: [] for loss_name in ["loss_eye", "val_loss_eye", "loss_pose_r", "val_loss_pose_r", "loss_au", "val_loss_au"]})
        
    def clear_loss(self):
        #clear the loss dict every epoch
        for key in self.log_dict.keys():
            self.log_dict[key].clear()
    
    def log_metrics_to_csv(self, metrics, file, mode="w"):
        # Log metrics to a CSV file
        file_path = join(constants.saved_path, file)

        try:
            with open(file_path, mode, newline="") as file:
                writer = csv.DictWriter(file, fieldnames=metrics.keys(), delimiter=";")
                if mode == "w":
                    writer.writeheader()

                writer.writerow(metrics)
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
    
