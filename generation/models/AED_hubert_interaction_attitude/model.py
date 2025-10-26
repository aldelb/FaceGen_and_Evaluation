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
from utils.data.data_utils import format_targets, reshape_output, concat_with_labels
from utils.data.noise_generator import NoiseGenerator
from utils.params_utils import save_params
from utils.evaluation.plot_utils import plotHistEpoch
from utils.model.model_parts import Down, OutConv, Up, Conv, ConvLayerNorm, DownLayerNorm
from torch.nn.utils import spectral_norm
from utils.data.labels import label_to_one_hot, one_hot_to_label



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
    
class BehavEmbeddingGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.in1_behav = Conv(constants.pose_size + constants.au_size, 32, constants.kernel_size) #100
        self.down1_behav = Down(32, 64, constants.kernel_size) #50
        self.down2_behav = Down(64, 128, constants.kernel_size) #25

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x1 = self.in1_behav(x)
        x2 = self.down1_behav(x1)
        x3 = self.down2_behav(x2)
        return x1, x2, x3
    
class PrevBehavEmbeddingGenerator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.in1_behav = Conv(constants.pose_size + constants.au_size, 32, constants.kernel_size) #12
        self.down1_behav = Down(32, 16, constants.kernel_size) #6
        self.down2_behav = Down(16, 8, constants.kernel_size) #3

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        x1 = self.in1_behav(x)
        x2 = self.down1_behav(x1)
        x3 = self.down2_behav(x2)
        return x3

class Generator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        bilinear = True
        factor = 2 if bilinear else 1
        self.noise_g = NoiseGenerator()

        ##encode inputs
        if constants.use_prev_audio:
            self.prev_audio_embedding = AudioEmbeddingGenerator()
        if constants.use_prev_behav:
            self.prev_behav_embedding = BehavEmbeddingGenerator()
        elif constants.use_small_prev_behav:
            self.prev_behav_embedding = PrevBehavEmbeddingGenerator()
        if constants.use_audio:
            self.audio_embedding = AudioEmbeddingGenerator()
        if constants.use_audio_speakerB:
            self.audio_embedding = AudioEmbeddingGenerator() #same weights as audio embedding?
        if constants.use_behav_speakerB:
            self.behav_embedding = BehavEmbeddingGenerator()
        
        ###concat with noise and labels here
        self.down4 = Down(constants.generator_weights["x3"], 512, constants.kernel_size) #12
        self.down5 = Down(512, 1024, constants.kernel_size) #6
        self.down6 = Down(1024, 2048 // factor, constants.kernel_size) #3

        ##Decoder eye
        self.up1_eye = Up(1024+1024+4+constants.x_small_prev_behav, 1024 // factor, constants.kernel_size, bilinear) #down6 + down5
        self.up2_eye = Up(512+512, 512 // factor, constants.kernel_size, bilinear) #up1 + down4
        self.up3_eye = Up(256+constants.generator_weights["x3"], 256 // factor, constants.kernel_size, bilinear) # up2 + x3
        self.up4_eye = Up(128+constants.generator_weights["x2"], 128 // factor, constants.kernel_size, bilinear) # up3 + x2
        self.up5_eye = Up(64+constants.generator_weights["x1"], 64, constants.kernel_size, bilinear) # up4 + x1
        self.outc_eye = OutConv(64, constants.eye_size, constants.kernel_size) #up5
 
        ##Decoder pose_r
        self.up1_pose_r = Up(1024+1024+4+constants.x_small_prev_behav, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_pose_r = Up(512+512, 512 // factor, constants.kernel_size, bilinear)
        self.up3_pose_r = Up(256+constants.generator_weights["x3"], 256 // factor, constants.kernel_size, bilinear)
        self.up4_pose_r = Up(128+constants.generator_weights["x2"], 128 // factor, constants.kernel_size, bilinear)
        self.up5_pose_r = Up(64+constants.generator_weights["x1"], 64, constants.kernel_size, bilinear)
        self.outc_pose_r = OutConv(64, constants.pose_r_size, constants.kernel_size)

        ##Decoder AUs
        self.up1_au = Up(1024+1024+4+constants.x_small_prev_behav, 1024 // factor, constants.kernel_size, bilinear)
        self.up2_au = Up(512+512, 512 // factor, constants.kernel_size, bilinear)
        self.up3_au = Up(256+constants.generator_weights["x3"], 256 // factor, constants.kernel_size, bilinear)
        self.up4_au = Up(128+constants.generator_weights["x2"], 128 // factor, constants.kernel_size, bilinear)
        self.up5_au = Up(64+constants.generator_weights["x1"], 64, constants.kernel_size, bilinear)
        self.outc_au = OutConv(64, constants.au_size, constants.kernel_size)


    def add_to_modalities(self, modalities, x1, x2, x3):
        modalities['x1'].append(x1)
        modalities['x2'].append(x2)
        modalities['x3'].append(x3)
        return modalities

    def forward(self, attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB):
        modalities = {"x1":[], "x2":[], "x3":[]}

        if constants.use_prev_audio:
            x1_prev_audio, x2_prev_audio, x3_prev_audio = self.prev_audio_embedding(prev_audio) #64, 128, 256
            modalities = self.add_to_modalities(modalities, x1_prev_audio, x2_prev_audio, x3_prev_audio)
        
        if constants.use_prev_behav:
            x1_prev_behav, x2_prev_behav, x3_prev_behav = self.prev_behav_embedding(prev_behav)
            modalities = self.add_to_modalities(modalities, x1_prev_behav, x2_prev_behav, x3_prev_behav)
        
        if constants.use_audio:
            x1_audio, x2_audio, x3_audio = self.audio_embedding(audio) #64, 128, 256
            modalities = self.add_to_modalities(modalities, x1_audio, x2_audio, x3_audio)
        
        if constants.use_audio_speakerB:
            x1_audio_speakerB, x2_audio_speakerB, x3_audio_speakerB = self.audio_embedding(audio_speakerB)
            modalities = self.add_to_modalities(modalities, x1_audio_speakerB, x2_audio_speakerB, x3_audio_speakerB)
        
        if constants.use_behav_speakerB:
            x1_behav_speakerB, x2_behav_speakerB, x3_behav_speakerB = self.behav_embedding(behav_speakerB)
            modalities = self.add_to_modalities(modalities, x1_behav_speakerB, x2_behav_speakerB, x3_behav_speakerB)

        x1 = torch.cat(modalities["x1"], 1) #64 + 64 + 64 + 32 + 32 = 256 (if all)
        x2 = torch.cat(modalities["x2"], 1) #128 + 128 + 128 + 64 + 64 = 512 (if all)
        x3 = torch.cat(modalities["x3"], 1) #256 + 256 + 256 + 128 + 128 = 1024 (if all)

        ###concat with noise and labels here
        noise = self.noise_g.getNoise(x3, std=constants.std_noise, interval=[-1,1])
        x3 = torch.add(x3, noise)


        #Encoder (audio + noise part)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        latent_representation = self.down6(x5)
        
        #add the attitudes to the latent representation
        redim_att = torch.repeat_interleave(attitudes.unsqueeze(2), latent_representation.shape[2], dim=2)
        latent_representation_att = torch.cat([latent_representation, redim_att], dim=1)

        #add prev_behav to the latent representation
        if constants.use_small_prev_behav:
            small_prev_behav = prev_behav[:,-12:,:] #take the last 12 frames of the previous behaviour
            embedding_small_prev_behav = self.prev_behav_embedding(small_prev_behav)
            latent_representation_att = torch.cat([latent_representation_att, embedding_small_prev_behav], dim=1)

        #Decoder gaze
        x = self.up1_eye(latent_representation_att, x5)
        x = self.up2_eye(x, x4)
        x = self.up3_eye(x, x3)
        x = self.up4_eye(x, x2)
        x = self.up5_eye(x, x1)
        logits_eye = self.outc_eye(x)
        logits_eye = torch.tanh(logits_eye)

        #Decoder pose_r
        x = self.up1_pose_r(latent_representation_att, x5)
        x = self.up2_pose_r(x, x4)
        x = self.up3_pose_r(x, x3)
        x = self.up4_pose_r(x, x2)
        x = self.up5_pose_r(x, x1)
        logits_pose_r = self.outc_pose_r(x)
        logits_pose_r = torch.tanh(logits_pose_r)

        #Decoder AUs
        x = self.up1_au(latent_representation_att, x5)
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
        x = torch.swapaxes(x, 1, 2)
        x = self.conv1_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv2_behaviour(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        return x

    
class Discriminator(pl.LightningModule):

    def __init__(self):
        super().__init__()

        if constants.use_prev_audio:
            self.prev_embed_audio = AudioEmbeddingDiscriminator()
        if constants.use_prev_behav:
            self.prev_embed_behaviour = BehavEmbeddingDiscriminator()
        if constants.use_audio:
            self.embed_audio = AudioEmbeddingDiscriminator()
        if constants.use_audio_speakerB:
            self.embed_audio = AudioEmbeddingDiscriminator() #same weights as audio embedding?
        self.embed_behaviour = BehavEmbeddingDiscriminator() #same embedding for second speaker

        self.conv_concat = ConvLayerNorm(constants.discriminator_weights + constants.number_of_dim_labels, 256, constants.kernel_size, constants.seq_len//4)
        self.conv_concat1 = ConvLayerNorm(256, 128, constants.kernel_size, constants.seq_len//4)
        self.conv_concat2 = ConvLayerNorm(128, 64, constants.kernel_size, constants.seq_len//4)
        self.fc1 = nn.utils.spectral_norm(nn.Linear(64 * floor(constants.seq_len//4), 64))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(64, 1))


    def forward(self, prev_audio, audio, prev_behav, current_behav, audio_speakerB, behav_speakerB):
        modalities = []
        if constants.use_prev_audio:
            prev_audio_embedding = self.prev_embed_audio(prev_audio)
            modalities.append(prev_audio_embedding)
        if constants.use_prev_behav:
            prev_behav_embedding = self.prev_embed_behaviour(prev_behav)
            modalities.append(prev_behav_embedding)
        if constants.use_audio:
            audio_embedding = self.embed_audio(audio)
            modalities.append(audio_embedding)

        behav_embedding = self.embed_behaviour(current_behav)
        modalities.append(behav_embedding)

        if constants.use_audio_speakerB:
            audio_speakerB_embedding = self.embed_audio(audio_speakerB)
            modalities.append(audio_speakerB_embedding)

        if constants.use_behav_speakerB:
            behav_speakerB_embedding = self.embed_behaviour(behav_speakerB)
            modalities.append(behav_speakerB_embedding)
        

        embedding = torch.cat(modalities, 1) #64*6 = 384

        x = self.conv_concat(embedding)
        x = self.conv_concat1(x)
        h_x = self.conv_concat2(x)
        x = h_x.view(h_x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x) #not adapted to the WGAN

        return x, h_x


class DiscriminatorAtt(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.conv_concat_att = ConvLayerNorm(64+4, 64, constants.kernel_size, constants.seq_len//4)
        self.fc1_att = nn.utils.spectral_norm(nn.Linear(64 * floor(constants.seq_len//4), 64))
        self.fc2_att = nn.utils.spectral_norm(nn.Linear(64, 1))


    def forward(self, h_x, attitudes, index_attitudes):
        attitudes, h_x = attitudes[index_attitudes], h_x[index_attitudes]

        #add the attitudes to h_x
        redim_att = torch.repeat_interleave(attitudes.unsqueeze(2), h_x.shape[2], dim=2)
        x_att = torch.cat([h_x, redim_att], dim=1)

        x_att = self.conv_concat_att(x_att)
        x_att = x_att.view(x_att.size(0), -1)
        x_att = self.fc1_att(x_att)
        x_att = self.fc2_att(x_att)
        return x_att
    


class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.c_lambda = constants.gp_weight
        self.evolution_fake = {"type":"", "number":0}
        self.criterion = nn.MSELoss()
        self.create_loss()

        self.automatic_optimization = False
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.discriminator_att = DiscriminatorAtt()

        self.generator.apply(self.weights_init_xavier)
        self.discriminator.apply(self.weights_init_xavier)
        self.discriminator_att.apply(self.weights_init_xavier)

        save_params(constants.saved_path, self.generator, self.discriminator)

    def forward(self, attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB):
        return self.generator(attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=constants.g_lr, betas=(constants.b1_g, 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=constants.d_lr, betas=(constants.b1_d, 0.9))
        d_opt_att = torch.optim.Adam(self.discriminator_att.parameters(), lr=constants.d_lr, betas=(constants.b1_d, 0.9))

        # Schedulers
        # g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_opt, gamma=1.0)  # no decay for G
        # d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_opt, gamma=0.98)  # decay progressively for D

        return g_opt, d_opt, d_opt_att
    
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
    
    def transition_smoothness_loss(self, prev_behav, generated_behav, real_behav):
        # reduce the difference between first frames of current and last frames of prev [same frames --> overlap]
        index_list = []
        for i in range(len(prev_behav)):
            if torch.allclose(prev_behav[i,-10:,], real_behav[i,:10,:], atol=1e-6) : #does not work with the beginning of a video (prev = 0, behav = normal)
                index_list.append(i) 
        smooth_loss = F.l1_loss(prev_behav[index_list,-10:,:], generated_behav[index_list,:10,:], reduction='none')
        smooth_loss = torch.mean(smooth_loss)
        return smooth_loss

    def mse_loss(self, gen, target):
        # Compute the MSE loss
        return self.criterion(gen, target).mean()
            
    def adversarial_loss(self, fake_pred):
        # Compute the generator loss using the discriminator's predictions
        # The generator tries to maximize the discriminator's predictions for fake samples
        return -1. * torch.mean(fake_pred)


    def generator_step(self, attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, index_attitudes, val=False):
            target_eye, target_pose_r, target_au = format_targets(behav)


            latent_representation, gen_eye, gen_pose_r, gen_au = self.generator(attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB)
            generated_current_behav = torch.cat((gen_eye, gen_pose_r, gen_au), 2)
            
            
            d_fake_pred, h_x = self.discriminator(prev_audio, audio, prev_behav, generated_current_behav, audio_speakerB, behav_speakerB)
            with torch.no_grad():
                d_real_pred, h_x = self.discriminator(prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB)

            d_fake_pred, d_real_pred = d_fake_pred.squeeze(), d_real_pred.squeeze()
            g_iteration_loss = self.adversarial_loss(d_fake_pred)

            if len(index_attitudes) > 0:
                d_fake_pred_att = self.discriminator_att(h_x, attitudes, index_attitudes).squeeze()
                g_iteration_loss_attitude = self.adversarial_loss(d_fake_pred_att)
            else:
                g_iteration_loss_attitude = torch.tensor(0.0)
            

            loss_eye = self.mse_loss(gen_eye, target_eye)
            loss_pose_r = self.mse_loss(gen_pose_r, target_pose_r)
            loss_au = self.mse_loss(gen_au, target_au)

            loss_temporal_smoothness = self.temporal_smoothness_loss(behav, generated_current_behav)
            loss_transition_smoothness = self.transition_smoothness_loss(prev_behav, generated_current_behav, behav)

            
            # Compute the generator loss
            g_loss = constants.adversarial_coeff * g_iteration_loss + constants.adversarial_coeff_att * g_iteration_loss_attitude + constants.eye_coeff * loss_eye + constants.pose_coeff * loss_pose_r + constants.au_coeff * loss_au + constants.temporal_smoothness * loss_temporal_smoothness + constants.transition_smoothness * loss_transition_smoothness

            # Log the losses
            if val:
                prefix = "val_"
            else:
                prefix = ""
            self.log_dict[prefix+"g_loss"].append(g_loss)
            self.log_dict[prefix+"g_iteration_loss"].append(constants.adversarial_coeff * g_iteration_loss)
            self.log_dict[prefix+"g_iteration_loss_att"].append(constants.adversarial_coeff_att * g_iteration_loss_attitude)
            self.log_dict[prefix+"loss_eye"].append(constants.eye_coeff * loss_eye)
            self.log_dict[prefix+"loss_pose_r"].append(constants.pose_coeff * loss_pose_r)
            self.log_dict[prefix+"loss_au"].append(constants.au_coeff * loss_au)
            self.log_dict[prefix+"temporal_smoothness"].append(constants.temporal_smoothness * loss_temporal_smoothness)
            self.log_dict[prefix+"transition_smoothness"].append(constants.transition_smoothness * loss_transition_smoothness)
            self.log_dict[prefix+"basic_real_pred"].append(torch.mean(d_real_pred))
            self.log_dict[prefix+"basic_fake_pred"].append(torch.mean(d_fake_pred))

            return latent_representation, g_loss
    

    def main_discriminator_step(self, attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, designed_batch):
            audio = audio.squeeze(1)

            with torch.no_grad():
                _, output_eye, output_pose_r, output_au = self.generator(attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB)
                generated_behav = torch.cat((output_eye, output_pose_r, output_au), 2)


            selected_prev_audio, selected_audio, selected_prev_behaviour, selected_real_behaviour, selected_fake_behaviour, selected_audio_speakerB, selected_behav_speakerB = self.create_fake_targets(prev_audio, audio, prev_behav, behav, generated_behav, audio_speakerB, behav_speakerB, designed_batch) 

            #fake predictions  (audio, prev_behav, generated_current_behav, audio_speakerB)
            fake_pred, _ = self.discriminator(selected_prev_audio, selected_audio, selected_prev_behaviour, selected_fake_behaviour, selected_audio_speakerB, selected_behav_speakerB)
            #real predictions 
            real_pred, _ = self.discriminator(selected_prev_audio, selected_audio, selected_prev_behaviour, selected_real_behaviour, selected_audio_speakerB, selected_behav_speakerB)

            fake_pred, real_pred = fake_pred.squeeze(), real_pred.squeeze()

            gp = self.compute_gradient_penalty(prev_audio, audio, prev_behav, behav, generated_behav, audio_speakerB, behav_speakerB)
            d_loss = torch.mean(fake_pred) - torch.mean(real_pred) +  self.c_lambda * gp

            # Log the losses
            self.log_dict["d_loss"].append(d_loss)
            self.log_dict["real_pred"].append(torch.mean(real_pred))
            self.log_dict["fake_pred"].append(torch.mean(fake_pred))
            self.log_dict["wasserstein_distance"].append(torch.mean(real_pred) - torch.mean(fake_pred))
            self.log_dict["penalty"].append(self.c_lambda * gp)

            return d_loss
    
    def create_fake_targets(self, prev_audio, audio, prev_behav, behav, generated_behav, audio_speakerB, behav_speakerB, designed_batch): 
        # Create the fake targets for the discriminator
        # The designed batch is a batch of generated examples with the some "mixed" real examples

        designed_criterias, designed_values = designed_batch
        d_prev_audio, d_audio, d_real_behaviour, d_prev_behaviour, d_fake_behaviour, d_audio_speakerB, d_behav_speakerB, _ = designed_values
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
        
        final_prev_audio = torch.cat((prev_audio[idx_generated], d_prev_audio[idx_designed]), dim=0)[new_order]
        final_audio = torch.cat((audio[idx_generated], d_audio[idx_designed]), dim=0)[new_order]
        final_prev_behaviour = torch.cat((prev_behav[idx_generated], d_prev_behaviour[idx_designed]), dim=0)[new_order]
        final_real_behaviour = torch.cat((behav[idx_generated], d_real_behaviour[idx_designed]), dim=0)[new_order]
        final_fake_behaviour = torch.cat((generated_behav[idx_generated], d_fake_behaviour[idx_designed]), dim=0)[new_order]
        final_audio_speakerB = torch.cat((audio_speakerB[idx_generated], d_audio_speakerB[idx_designed]), dim=0)[new_order]
        final_behav_speakerB = torch.cat((behav_speakerB[idx_generated], d_behav_speakerB[idx_designed]), dim=0)[new_order]

        return final_prev_audio, final_audio, final_prev_behaviour, final_real_behaviour, final_fake_behaviour, final_audio_speakerB, final_behav_speakerB
    
    def compute_gradient_penalty(self, prev_audio, audio, prev_behav, real_samples, fake_samples, audio_speakerB, behav_speakerB):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(len(real_samples), 1, 1, requires_grad=True).to(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(real_samples)
        d_interpolates, _ = self.discriminator(prev_audio, audio, prev_behav, interpolates, audio_speakerB, behav_speakerB) #real inputs

        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates), 
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]
        
        gradient = gradient.reshape(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        self.log("grad_norm_penalty", gradient_norm.mean())  # log the gradient norm
        penalty = torch.mean((gradient_norm - 1)**2).mean()

        return penalty
    

    def att_discriminator_step(self, attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, index_attitudes):
        audio = audio.squeeze(1)

        with torch.no_grad():
            _, output_eye, output_pose_r, output_au = self.generator(attitudes, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB)
            generated_behav = torch.cat((output_eye, output_pose_r, output_au), 2)
        
        #fake predictions 
        _, fake_h_x = self.discriminator(prev_audio, audio, prev_behav, generated_behav, audio_speakerB, behav_speakerB)
        fake_pred_att = self.discriminator_att(fake_h_x, attitudes, index_attitudes).squeeze() #TODO: faire des faux exemples ?
        #real predictions
        _, real_h_x = self.discriminator(prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB)
        real_pred_att = self.discriminator_att(real_h_x, attitudes, index_attitudes).squeeze()


        gp_att = self.compute_gradient_penalty_att(real_h_x, fake_h_x, attitudes, index_attitudes)
        d_loss_att = torch.mean(fake_pred_att) - torch.mean(real_pred_att) + self.c_lambda * gp_att

        # Log the losses
        self.log_dict["d_loss_att"].append(d_loss_att)
        
        return d_loss_att
        
    

    def compute_gradient_penalty_att(self, real_samples, fake_samples, attitudes, index_attitudes):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(len(real_samples), 1, 1, requires_grad=True).to(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(real_samples)
        d_interpolates = self.discriminator_att(interpolates, attitudes, index_attitudes) #real inputs

        gradient = torch.autograd.grad(
            inputs=interpolates,
            outputs=d_interpolates,
            grad_outputs=torch.ones_like(d_interpolates), 
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]
        
        gradient = gradient.reshape(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        self.log("grad_norm_penalty_att", gradient_norm.mean())  # log the gradient norm
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

        prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, attitudes = real_batch[0], real_batch[1], real_batch[2], real_batch[3], real_batch[4], real_batch[5], real_batch[6]


        g_opt, d_opt, d_opt_att = self.optimizers()

        None_one_hot = label_to_one_hot("None", "small_attitude").to(attitudes)
        match_mask = ~torch.all(attitudes == None_one_hot, dim=1)
        index_attitudes = torch.where(match_mask)[0]

        for i in range(constants.n_critics):
            d_loss = self.main_discriminator_step(attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, designed_batch[i])

            d_opt.zero_grad()
            self.manual_backward(d_loss)
            self.log_gradients(self.discriminator, "discriminator")
            d_opt.step()
            
            if len(index_attitudes) > 0:
                d_loss_att = self.att_discriminator_step(attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, index_attitudes)
                d_opt_att.zero_grad()
                self.manual_backward(d_loss_att)
                self.log_gradients(self.discriminator_att, "discriminator_att")
                d_opt_att.step()

        _, g_loss = self.generator_step(attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, index_attitudes)
        
        self.log("lr_d", d_opt.param_groups[0]["lr"])
        self.log("lr_g", g_opt.param_groups[0]["lr"])
        self.log("lr_d_att", d_opt_att.param_groups[0]["lr"])
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.log_gradients(self.generator, "generator")
        g_opt.step()
            

    def adjuste_designed_examples(self, dict_loss, pourcent_generated):
        # Adjuste the percentage of generated examples in the designed batch (if tendency is observed during 5 epochs)
        if dict_loss["avg_basic_real_pred"] < dict_loss["avg_basic_fake_pred"]:
            if self.evolution_fake["type"] == "+" : 
                self.evolution_fake["number"] += 1
                if self.evolution_fake["number"] >= 5 :
                    pourcent_generated = pourcent_generated + constants.step_pourcent
                    self.evolution_fake["number"] = 0
            else: 
                self.evolution_fake["type"] = "+"
                self.evolution_fake["number"] = 1


        elif dict_loss["avg_fake_pred"] < dict_loss["avg_basic_fake_pred"]:
            if self.evolution_fake["type"] == "-" : 
                self.evolution_fake["number"] += 1
                if self.evolution_fake["number"] >= 5 :
                    pourcent_generated = pourcent_generated - constants.step_pourcent
                    self.evolution_fake["number"] = 0
            else: 
                self.evolution_fake["type"] = "-"
                self.evolution_fake["number"] = 1
        
            
        
        pourcent_generated = max(min(1, pourcent_generated),constants.initial_pourcent_generated)
        print(dict_loss["avg_basic_real_pred"], dict_loss["avg_basic_fake_pred"], dict_loss["avg_fake_pred"], pourcent_generated, self.evolution_fake)
        
        return pourcent_generated


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

        if(constants.evolution_pourcent_generated):
            constants.pourcent_generated = self.adjuste_designed_examples(dict_loss, constants.pourcent_generated)

        _save_losses(dict_loss)
        self.clear_loss()


    def validation_step(self, batch, batch_idx):
        # Validation step
        # This function is called during validation to compute the validation loss
        prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, attitudes = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        
        None_one_hot = label_to_one_hot("None", "small_attitude").to(attitudes)
        match_mask = ~torch.all(attitudes == None_one_hot, dim=1)
        index_attitudes = torch.where(match_mask)[0]

        with torch.no_grad():
            _, val_g_loss = self.generator_step(attitudes, prev_audio, audio, prev_behav, behav, audio_speakerB, behav_speakerB, index_attitudes, val=True)

        return val_g_loss




    ################ Log/Loss processing #########################
    def create_loss(self):
        #create the loss dict for the training
        self.log_dict = {loss_name: [] for loss_name in ["g_loss", "val_g_loss", "g_iteration_loss", "val_g_iteration_loss", "g_iteration_loss_att", "val_g_iteration_loss_att", "d_loss", "d_loss_att", "penalty", "wasserstein_distance", "temporal_smoothness", "val_temporal_smoothness", "transition_smoothness", "val_transition_smoothness"]}
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
    
