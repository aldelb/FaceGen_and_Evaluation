import pandas as pd
import pytorch_lightning as pl
import torch
import os
from os.path import join, isfile
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.data import data_utils
import utils.constants.features as features
from utils.data.data_utils import scale_from_scratch, scale
from utils.data.labels import label_to_one_hot
import utils.constants.constants as constants
import utils.data.noise_generator as noise_generator


class CustomDataset(Dataset):
    def __init__(self, keys=None, details_time=None, behav_scaled=None, behav=None, locuteur=None, role=None, attitude=None, gender=None, all_keys=None, final_behaviour=None):
        self.keys = keys
        self.details_time = details_time
        self.behav_scaled = behav_scaled
        self.behav = behav
        self.locuteur = locuteur
        self.role = role
        self.attitude = attitude
        self.gender = gender
        self.all_keys = all_keys
        self.final_behaviour = final_behaviour


    def __len__(self):
        return len(self.behav)
    
    def get_behav_scaled(self):
        return self.behav_scaled
    
    def get_behav(self, noise_name=None):
        print(self.behav.shape)
        #on fait un tensor vide de la forme de behav
        if noise_name is not None:
            noisy_behav = noise_generator.get_noise_torch(self.behav, noise_name)
            if noise_name == "mismatched":
                noisy_behav = noisy_behav[torch.randperm(noisy_behav.shape[0])]
            print("noisy_behav", noisy_behav.shape)
            return noisy_behav
        return self.behav
    
    def get_keys(self):
        return self.keys
    
    def get_details_time(self):
        return self.details_time
    
    def get_locuteur(self):
        return self.locuteur
    
    def get_role(self):
        return self.role
    
    def get_attitude(self):
        return self.attitude
    
    def get_gender(self):
        return self.gender
    
    def get_all_keys(self):
        return self.all_keys
    
    def get_final_behaviour(self):
        return self.final_behaviour
    
    def get_behaviour_by_key(self, key):
        if key in self.all_keys:
            index = self.all_keys.index(key)
            return self.final_behaviour[index]
        else:
            raise ValueError(f"Key {key} not found in dataset.")

    def __getitem__(self, i):
        base_output = self.keys[i], self.behav_scaled[i], self.gender[i], self.role[i], self.attitude[i]
        return base_output
            

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, set_type, data_type, epoch=None):
        super(CustomDataModule, self).__init__()
        self.is_prepared = False
        self.set_type = set_type
        self.data_type = data_type
        self.epoch = epoch
        self.interaction = False

    def init_data_dict(self):
        data_dict = {"keys": [], "details_time": [], "behaviour": [],
                     "locuteur": [], "attitude": [], "role": [], "gender": [],
                     "all_keys": [], "final_behaviour": []}
        return data_dict
    
    def load_data(self):
        final_dict = self.init_data_dict()
        data_path = constants.data_path        
        details_file = join(data_path, "details_global.xlsx")
        details = pd.read_excel(details_file)
        list_of_keys = details[["nom", "set"]].where(details["set"] == self.set_type).dropna()["nom"].values.tolist()
        print(list_of_keys)
        for key in list_of_keys:
            dataset = details[details["nom"] == key]["part"].values[0]

            if(self.data_type =="real"):
                extended_data_path = join(data_path, dataset, "final_data")
                if("interaction" in constants.model_name):
                    self.interaction = True

            elif(self.data_type == "generated"):
                extended_data_path = join(constants.output_path, constants.model_path, "epoch_" + str(self.epoch), self.set_type)

            path_file = join(extended_data_path, key + ".csv")
            if(isfile(path_file)):
                print("Loading file: ", key, "in dataset: ", dataset, "for set: ", self.set_type)
                locuteur = details[details["nom"] == key]["locuteur"].values[0]
                attitude = details[details["nom"] == key]["attitude"].values[0]
                role = details[details["nom"] == key]["role"].values[0]
                genre = details[details["nom"] == key]["genre"].values[0]

                final_dict = self.load_file_data(path_file, key, final_dict, locuteur, attitude, role, genre)

        return self.format_dict(final_dict)


    def load_file_data(self, csv_file, key, final_dict, locuteur, attitude, role, genre):
        df = pd.read_csv(csv_file)
        print(self.data_type, "data loaded")
        if(self.data_type == "real"):
            if(self.interaction):
                df = data_utils.putListeningToZero(df, ["AU25_r", "AU26_r"])
            else:
                df = data_utils.putListeningToZero(df, features.OPENFACE_FEATURES)

            
        final_dict["all_keys"].append(key)
        final_dict["final_behaviour"].append(df[features.OPENFACE_FEATURES])

        t1, t2 = 0, features.SEGMENT_LENTGH
        end_time_result = df["timestamp"].iloc[-1]
        while t2 <= end_time_result :
            cut = df[(df["timestamp"] < t2) & (df["timestamp"] >= t1)]
            final_dict["details_time"].append(cut["timestamp"].values)
            final_dict["behaviour"].append(cut[features.OPENFACE_FEATURES].values)

            final_dict["keys"].append(key)
            final_dict["locuteur"].append(locuteur)
            final_dict["attitude"].append(attitude)
            final_dict["role"].append(role)
            final_dict["gender"].append(genre)
            
            t1, t2 = round(t1 + features.SEGMENT_LENTGH - features.OVERLAP,2), round(t2 + features.SEGMENT_LENTGH - features.OVERLAP,2)

        return final_dict


    def format_dict(self, dict):
        keys = dict["keys"]
        details_time = torch.as_tensor(np.array(dict["details_time"]))
        behaviour = torch.as_tensor(np.array(dict["behaviour"]))
        locuteur = dict["locuteur"]
        attitude = torch.stack([label_to_one_hot(label, "attitude") for label in dict["attitude"]])
        role = torch.stack([label_to_one_hot(label, "role") for label in dict["role"]])
        gender = torch.stack([label_to_one_hot(label, "gender") for label in dict["gender"]])
        
        constants.seq_len = behaviour.shape[1]
        constants.behav_dim = behaviour.shape[2]

        return keys, details_time, behaviour, locuteur, attitude, role, gender, dict["all_keys"], dict["final_behaviour"]


    def prepare_data(self):
        if not self.is_prepared:
            print("Lauching of prepare_data")   
            train_data = self.load_data()
            self.keys = train_data[0]
            self.details_time = train_data[1]
            self.behav = train_data[2]
            self.locuteur = train_data[3]
            self.attitude = train_data[4]
            self.role = train_data[5]
            self.gender = train_data[6]
            self.all_keys = train_data[7]
            self.final_behaviour = train_data[8]


    def setup(self):
        if not self.is_prepared:   
            dir_scaler = os.path.join(constants.dir_path, "generation", "utils", "model") #scaler of the autoencoder_AUFGD
            y_scaler = pickle.load(open(join(dir_scaler, 'scaler_y.pkl'), 'rb'))
            
            behav_scaled = scale(self.behav, y_scaler)
            self.dataset = CustomDataset(keys=self.keys, details_time=self.details_time,
                                                behav_scaled=behav_scaled,
                                                behav = self.behav,
                                                locuteur=self.locuteur, attitude=self.attitude, role=self.role,
                                                gender=self.gender,
                                                all_keys=self.all_keys, final_behaviour=self.final_behaviour)


            
            self.is_prepared = True