import os
import pandas as pd
from os.path import isdir
import time

import utils.constants.constants as constants
from utils.logger import MyLogger


class Generate():
    def __init__(self, args, set_type):
        super(Generate, self).__init__()

        epoch = int(args.epoch)
        self.file = args.file

        constants.number_of_gpu = int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_NNODES'])

        logger = MyLogger()
        trainer_args = self._create_trainer_args(logger)
        dm = self._prepare_data_module(set_type)
        model = self._load_model(epoch, dm)
        output_path = self._prepare_output_path(epoch, set_type)
        self._generate_and_save_motion(model, trainer_args, dm, set_type, output_path)


    def _create_trainer_args(self, logger):
        return {
            'accelerator': 'gpu',
            "enable_progress_bar": False,
            "logger": logger
        }

    def _prepare_data_module(self, set_type):
        dm = constants.customDataModule(file=self.file, stage="predict", set=set_type)
        dm.prepare_data()
        dm.setup(stage="predict")
        return dm

    def _load_model(self, epoch, dm):
        checkpoint_path = os.path.join(constants.saved_path, f"epoch={epoch - 1}.ckpt")
        model = constants.model.load_from_checkpoint(checkpoint_path)
        model.pose_scaler = dm.y_scaler
        return model

    def _prepare_output_path(self, epoch, set_type):
        path = os.path.join(constants.output_path, constants.model_path, f"epoch_{epoch}", set_type)
        os.makedirs(path, exist_ok=True)
        return path

    def _generate_and_save_motion(self, model, trainer_args, dm, set_type, output_path):
        start_time = time.time()
        all_generated_df = constants.generate_motion(model, trainer_args, dm)

        for key, df_list in all_generated_df.items():
            #save_key = self.file if self.file else key
            self._save_final_file(key, df_list, output_path)

        end_time = time.time()
        print(f"Motion generation completed in {end_time - start_time} seconds for {set_type}.")
        print("All files generated in", output_path)

    def _save_final_file(self, filename, df_list, output_path):
        save_file = os.path.join(output_path, f"{filename}.csv")
        df = self._get_final_file(df_list, filename)
        df.to_csv(save_file)
        return df
    
    def _get_final_file(self, df_list, filename):
        df = pd.concat(df_list, ignore_index=True)
        #recuprer les timestamp égaux et calculer la distance moyenne entre eux pour ceux qui sont égaux
        average_dist = {col: 0.0 for col in df.columns}
        nb = 0
        for timestamp in df['timestamp'].unique():
            subset = df[df['timestamp'] == timestamp]
            if len(subset) > 1:
                #calculer la distance moyenne entre les points pour chaque colonne, il y en a deux
                dist = (subset.iloc[-1] - subset.iloc[0]).to_dict()
                #toutes les valeurs en absolu
                dist = {col: abs(val) for col, val in dist.items()}
                for col in average_dist.keys():
                    average_dist[col] += dist[col]
                nb += 1
        if nb > 0:
            average_dist = {col: val / nb for col, val in average_dist.items()}
        print("average distance between points for same timestamp:", average_dist)
        #moyenne globale des distances du dict
        global_average = sum(average_dist.values()) / len(average_dist) if average_dist else 0
        print(filename, global_average)

        df = df.groupby('timestamp').mean().reset_index()
        
        df.set_index("timestamp", inplace=True)
        return df
        