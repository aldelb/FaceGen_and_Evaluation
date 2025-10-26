import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import utils.constants.constants as constants
import utils.constants.features as features

class GenerateModel():
    def __init__(self):
        super(GenerateModel, self).__init__()


    def generate_motion(self, model, trainer_args, dm):
        #all predictions
        trainer = pl.Trainer(**trainer_args)
        predictions = trainer.predict(model, dm)

        #create the df for final files
        all_files = {}
        for keys, preds, details_times, _, _ in predictions: #all batches
            for index, key in enumerate(keys): #batch size
                pred = preds[index]
                details_time = details_times[index]
                if(key not in all_files.keys()):
                    all_files[key] = []
                out = np.concatenate((np.array(details_time).reshape(-1,1), pred[:,:constants.eye_size], np.zeros((pred.shape[0], 3)), pred[:,constants.eye_size:]), axis=1)
                df = pd.DataFrame(data = out, columns = features.OUTPUT_COLUMNS)
                all_files[key].append(df)

        return all_files
    
    def generate_latent(self, model, trainer_args, dm):
        #all predictions
        trainer = pl.Trainer(**trainer_args)
        predictions = trainer.predict(model, dm)

        #create the df for final files
        all_files = {}
        for keys, _, details_times, latents, _ in predictions: #all batches
            for index, key in enumerate(keys): #batch size
                latents_values = latents[index]
                if(key not in all_files.keys()):
                    all_files[key] = []
                all_files[key].append(latents_values)

        return all_files
