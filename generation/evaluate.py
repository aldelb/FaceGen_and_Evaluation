from math import ceil
import os
from os.path import isdir, isfile, join
import pickle
import time
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from dtaidistance import dtw
from torch.utils.data import DataLoader
import torch

import utils.constants.constants as constants
import utils.evaluation.metrics.raw_metrics as raw_metrics
import utils.evaluation.metrics.latents_metrics as latents_metrics
import utils.evaluation.csv_to_dataset as csv_to_dataset
import utils.model.autoencoder_AUFGD as genea_ae
import utils.constants.features as features
import utils.data.noise_generator as noise_generator


class Evaluate():
    def __init__(self, set_type, args, ref=False, noise=False):
        super(Evaluate, self).__init__()
        #---------------------------------------------------
        # We never call the model in the evaluation, so we don't need to load it, we just need the csv files
        #---------------------------------------------------

        self.real_frames = None
        self.generated_frames = None
        self.latent_encoder = None
        
        dm_real = csv_to_dataset.CustomDataModule(set_type=set_type, data_type="real")
        dm_real.prepare_data()
        dm_real.setup()
        real_dataset = dm_real.dataset

        # Use the real dataset as generated dataset
        # suppose to obtain the best results
        # need another idea for the ref
        if(ref): 
            print("\n", "Evaluation ref", "on", set_type)
            generated_dataset = real_dataset

            path_evaluation = self._create_ref_path(set_type, "ref")
            print("Evaluation path:", path_evaluation)
            
            self._launch_eval(args, real_dataset, generated_dataset, path_evaluation)

        # Add noise to the real dataset and use it as generated dataset
        elif(noise):
            print("\n", "Evaluation ref", "on", set_type, "dataset with noise", args.noise_name)
            generated_dataset = real_dataset

            path_evaluation = self._create_ref_path(set_type, args.noise_name)
            print("Evaluation path:", path_evaluation)

            self._launch_eval(args, real_dataset, generated_dataset, path_evaluation, args.noise_name)

        # Load the generated files from the model we evaluate
        else: 
            epoch = int(args.epoch)

            dm_generated = csv_to_dataset.CustomDataModule(set_type=set_type, data_type="generated", epoch=epoch)
            dm_generated.prepare_data()
            dm_generated.setup()
            generated_dataset = dm_generated.dataset

            path_evaluation = self._create_path(epoch, set_type)

            print("Evaluation of the model", constants.model_path, "at epoch", epoch, "on", set_type, "dataset")
            print("Evaluation path:", path_evaluation)

            self._launch_eval(args, real_dataset, generated_dataset, path_evaluation)
    

    

    #---------------------------------------------------
    # Paths creation
    #---------------------------------------------------
    def _create_ref_path(self, set_type, name):
        path = join(constants.evaluation_path, f"ref_{name}", set_type)
        os.makedirs(path, exist_ok=True)
        return path
    
    def _create_path(self, epoch, set_type):
        path = join(constants.evaluation_path, constants.model_path, f"epoch_{epoch}", set_type)
        os.makedirs(path, exist_ok=True)
        return path


    #---------------------------------------------------
    # Data loading
    #---------------------------------------------------

    
    def _getSegmentVideoData(self, key, dataset, noise_name=None):
        #evaluate on 30 sec segment to be able to compute the significance of the metrics
        #we cut the video into 1 min segments
        all_df_segments = []
        df_data = dataset.get_behaviour_by_key(key)
        #cut into 1 min segment --> len = 1500 frames
        for i in range(0, len(df_data), 1500):
            # print("Segment", i, "to", i+1500, "of video", key)
            df_segment = df_data.iloc[i:i+1500]
            if(noise_name is not None):
                df_segment = noise_generator.get_noise_df(df_segment, noise_name)
            all_df_segments.append(df_segment.copy())
        
        if(noise_name == "mismatched"):
            # shuffle the segments
            np.random.shuffle(all_df_segments)

        return all_df_segments
    
    def _getDictPerSegmentData(self, dataset, noise_name=None):
        dict = {}
        for key in dataset.get_all_keys():
            dict[key] = self._getSegmentVideoData(key, dataset, noise_name)
        return dict
    
    def _getVideoData(self, key, dataset, noise_name=None):
        df_data = dataset.get_behaviour_by_key(key)
        if(noise_name is not None):
            noisy_data = noise_generator.get_noise_df(df_data, noise_name)
            return noisy_data
        
        return df_data
    
    def _getDictPerVideoData(self, dataset, noise_name=None):
        dict = {}
        for key in dataset.get_all_keys():
            dict[key] = self._getVideoData(key, dataset, noise_name)
        return dict

    def _getDataFrames(self, dataset, noise_name=None):
        all_frames = pd.DataFrame(columns=features.ALL_FEATURES)
        for key in dataset.get_all_keys():
            current_frames = self._getVideoData(key, dataset, noise_name)
            all_frames = pd.concat([all_frames, current_frames], ignore_index=True)
        return all_frames
    

    
    def _load_latent_encoder(self):
        checkpoint_dir_path = os.path.join(constants.dir_path, "generation", "utils", "model", "latent_space")

        if("interaction" in constants.model_name):
            checkpoint_dir_path = os.path.join(checkpoint_dir_path, "interaction")
            print("Load latent encoder for interaction model")
        else: 
            checkpoint_dir_path = os.path.join(checkpoint_dir_path, "base")
            print("Load latent encoder for base model")

        checkpoint_file= sorted(
                [f for f in os.listdir(checkpoint_dir_path) if f.endswith('.ckpt')],
                key=lambda f: float(f.split('val_loss=')[-1].replace('.ckpt', '')))[0]
        
        self.latent_encoder = genea_ae.GeneaConvAutoEncoder.load_from_checkpoint(os.path.join(checkpoint_dir_path, checkpoint_file), map_location=torch.device('cpu'))


    def _getLatentData(self, dataset, noise_name=None):
        if(self.latent_encoder is None):
            self._load_latent_encoder()

        self.latent_encoder.eval()
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        latents = []
        for batch in loader:
            x = batch[1]
            if(noise_name is not None):
                x = noise_generator.get_noise_torch(x, noise_name)
            z = self.latent_encoder.extract_latent(x)
            latents.append(z)
        latents = torch.cat(latents, dim=0).cpu().numpy()

        if(noise_name == "mismatched"):
            # shuffle the latents
            print("Shuffle latents")
            latents = latents[torch.randperm(latents.shape[0])]
        return latents


    #---------------------------------------------------
    # Evaluation methods
    #---------------------------------------------------
    
    def _launch_eval(self, args, real_dataset, generated_dataset, path_evaluation, noise_name=None):
        #Calculate the execution time between two points
        start_time = time.time()

        if(args.raw):
            self._evaluate_raw_data(real_dataset, generated_dataset, path_evaluation, noise_name)

        if(args.latent):
            self._evaluate_latent_data(real_dataset, generated_dataset, path_evaluation, noise_name)

        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time} seconds.")

        if(args.visu):
            self._visu_data(real_dataset, generated_dataset, path_evaluation, noise_name)

    def _evaluate_raw_data(self, real_dataset, generated_dataset, path_evaluation, noise_name):
        dict_real = self._getDictPerSegmentData(real_dataset)
        dict_generated = self._getDictPerSegmentData(generated_dataset, noise_name)

        raw_metrics.compute_motion_measures(dict_real, dict_generated, path_evaluation)
        raw_metrics.compute_hellinger_distance(dict_real, dict_generated, path_evaluation)
        raw_metrics.create_dtw(dict_real, dict_generated, path_evaluation)
        raw_metrics.create_wpd(real_dataset.get_behav(), generated_dataset.get_behav(noise_name), path_evaluation)
        raw_metrics.create_dice_score_and_coverage(dict_real, dict_generated, path_evaluation)
        raw_metrics.create_range_validity(dict_real, dict_generated, path_evaluation)

    
    def _visu_data(self, real_dataset, generated_dataset, path_evaluation, noise_name=None):
        # Not mixed for the noise "mismatched"
        dict_real = self._getDictPerVideoData(real_dataset)
        dict_generated = self._getDictPerVideoData(generated_dataset, noise_name)

        real_frames = self._getDataFrames(real_dataset, noise_name)
        generated_frames = self._getDataFrames(generated_dataset, noise_name)
    
        raw_metrics.create_pca(real_frames, generated_frames, path_evaluation)
        raw_metrics.create_curve(real_frames, generated_frames, path_evaluation)
        raw_metrics.create_curves_video(dict_real, dict_generated, path_evaluation)

    
    def _evaluate_latent_data(self, real_dataset, generated_dataset, path_evaluation, noise_name=None):
        latent_real = self._getLatentData(real_dataset)
        latent_generated = self._getLatentData(generated_dataset, noise_name)
        df = pd.DataFrame(columns=["Metric", "Value"])
        df["Metric"] = [
            "FGD_real",
            "FGD_generated",
            "Coverage_real",
            "Coverage_generated",
            "MMS_real",
            "MMS_generated",
            "APD_real",
            "APD_generated",
            "Density_real",
            "Density_generated",
        ]

        fgd_real = latents_metrics.compute_fgd(latent_real, latent_real)
        fgd_generated = latents_metrics.compute_fgd(latent_real, latent_generated)
        df.loc[df["Metric"] == "FGD_real", "Value"] = fgd_real
        df.loc[df["Metric"] == "FGD_generated", "Value"] = fgd_generated

        coverage_real = latents_metrics.compute_coverage(latent_real, latent_real)
        coverage_generated = latents_metrics.compute_coverage(latent_real, latent_generated)
        df.loc[df["Metric"] == "Coverage_real", "Value"] = coverage_real
        df.loc[df["Metric"] == "Coverage_generated", "Value"] = coverage_generated
        
        mms_real = latents_metrics.compute_mms(latent_real, latent_real)
        mms_generated = latents_metrics.compute_mms(latent_real, latent_generated)
        df.loc[df["Metric"] == "MMS_real", "Value"] = mms_real
        df.loc[df["Metric"] == "MMS_generated", "Value"] = mms_generated

        apd_real = latents_metrics.compute_apd(latent_real, Sapd=100, runs=5, seed=42)
        apd_generated = latents_metrics.compute_apd(latent_generated, Sapd=100, runs=5, seed=42)
        df.loc[df["Metric"] == "APD_real", "Value"] = apd_real
        df.loc[df["Metric"] == "APD_generated", "Value"] = apd_generated

        density_real = latents_metrics.compute_density(latent_real, latent_real)
        density_generated = latents_metrics.compute_density(latent_real, latent_generated)
        df.loc[df["Metric"] == "Density_real", "Value"] = density_real
        df.loc[df["Metric"] == "Density_generated", "Value"] = density_generated

        df.to_excel(path_evaluation + "/latents_metrics.xlsx", index=False)

        