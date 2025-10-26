import pandas as pd
import pytorch_lightning as pl
import torch
import os
from os.path import join, isdir, isfile
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset

import utils.constants.constants as constants
from utils.data.data_utils import scale_from_scratch, scale
from utils.data.labels import label_to_one_hot, get_maj_label
from utils.data.designedDataset import DesignedDataset1
from utils.data.hubert_embedding import HubertEmbedding


class CustomDataset(Dataset):
    def __init__(self, keys=None, X_audio=None, Y=None, speak_or_not=None, labels=None, details_time=None, final_Y=None, predict=False, final_keys=None):
        self.X_audio = X_audio
        self.Y = Y
        self.speak_or_not = speak_or_not
        self.labels = labels
        self.details_time = details_time
        self.final_Y = final_Y
        self.predict = predict
        self.keys = keys
        self.final_keys = final_keys

    def __len__(self):
        return len(self.X_audio)

    def __getitem__(self, i):
        if(self.predict):
            base_output = self.X_audio[i], self.details_time[i], self.keys[i]
            for label_type in constants.main_list_for_loading:
                base_output += (self.labels[label_type][i],)
            return base_output
        
        else:
            base_output = self.X_audio[i], self.Y[i]
            for label_type in constants.main_list_for_loading:
                base_output += (self.labels[label_type][i],)
        return base_output

    
    def get_final_videos(self):
        return {"key": self.final_keys, "video": self.final_Y}
            

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, fake_examples = None, file=None, stage=None, set=None):
        super(CustomDataModule, self).__init__()
        self.is_prepared = False
        self.batch_size = constants.batch_size
        if(fake_examples == None):
            self.fake_examples = constants.fake_target
        else:
            self.fake_examples = fake_examples
        
        self.stage = stage
        self.set = set
        self.file = file

    def init_data_dict(self):
        data_dict = {"keys": [],
            "prev_X_audio": [], "X_audio": [], "prev_behaviour": [], "behaviour": [],
            "prev_details_time": [], "details_time": [], 
            "prev_speak_or_not": [], "speak_or_not": [],
            "final_behaviour": [], "final_key": []}
        
        for type in constants.main_list_for_loading:
            data_dict[type] = []

        return data_dict

    
    def load_data(self, set_type):
        final_dict = self.init_data_dict()
        hubert_embedding_extractor = HubertEmbedding()
        data_path = constants.data_path
        datasets_properties = constants.datasets_properties
    
        details_file = join(data_path, "details_global.xlsx")
        details = pd.read_excel(details_file)
        list_of_keys = details[["nom", "set"]].where(details["set"] == set_type).dropna()["nom"].values.tolist()

        for key in list_of_keys:
            dataset = details[details["nom"] == key]["part"].values[0]
            path_file = join(data_path, dataset, "final_data", datasets_properties, key + ".p")
            if(isfile(path_file)):
                print("Loading file: ", key, "in dataset: ", dataset, "for set: ", set_type)
                final_dict = self.load_file_data(path_file, final_dict, hubert_embedding_extractor)

        return self.format_dict(final_dict)

    def load_one_file(self, file):
        final_dict = self.init_data_dict()
        hubert_embedding_extractor = HubertEmbedding()
        path_file = join(constants.data_path, "audio_tts", "final_data", constants.datasets_properties, file + ".p")
        
        final_dict = self.load_file_data(path_file, final_dict, hubert_embedding_extractor, one_file=True)
        return self.format_dict(final_dict, one_file=True)


    def load_file_data(self, path_file, final_dict, hubert_embedding_extractor, one_file=False):
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        #si "hubert" n'est pas une clé de data, on l'ajoute
        if "hubert" not in data.keys():
            data["hubert"] = {}

        if constants.hidden_state_index not in data["hubert"].keys():
            print("The file does not contain hubert embedding, creation...")
            audio_data = torch.stack(data["wav2vec"], 0)
            hubert_embedding = hubert_embedding_extractor.create_hubert_embedding(audio_data.squeeze(1))
            data["hubert"][constants.hidden_state_index] = hubert_embedding
            pickle.dump(data, open(path_file, 'wb'))
            print("The file has been updated with hubert embedding")
        else:
            hubert_embedding = data["hubert"][constants.hidden_state_index]
        
        final_dict["X_audio"].extend(hubert_embedding)

        
        final_dict["keys"].extend([ele for ele in [data["key"]] for _ in range(len(data["speak_or_not"]))])
        final_dict["details_time"].extend(data["details_time"])
        final_dict["speak_or_not"].extend(data["speak_or_not"])

        for type in constants.main_list_for_loading:
            if(type in ["gender", "attitude", "role"]):
                final_dict[type].extend([ele for ele in [data[type]] for _ in range(len(data["speak_or_not"]))])
            else:
                final_dict[type].extend([get_maj_label(ele) for ele in data[type]])
        
        if(one_file):
            final_dict["behaviour"].extend([])
            final_dict["final_behaviour"].append(None)
        else:
            final_dict["behaviour"].extend(data["behaviour_listening_to_zero"])
            final_dict["final_behaviour"].append(data["final_behaviour_listening_to_zero"])

        final_dict["final_key"].append(data["key"])

        del data # Free up memory occupied by the 'data' variable

        return final_dict

    def format_dict(self, dict, one_file=False):

        speak_or_not = torch.as_tensor(np.array(dict["speak_or_not"], dtype=np.float32)) #TODO: OK ?
        speak_or_not_extended = torch.repeat_interleave(speak_or_not, 2, dim=1).unsqueeze(-1)
        audio = torch.stack(dict["X_audio"] , 0).squeeze()
        audio = torch.cat((speak_or_not_extended, audio), dim=2)

        one_hot_tensor_list = {}
        for type in constants.main_list_for_loading:
            if(type in ["gender"]):
                updated_gender = [dict["gender"][i] if np.mean(np.array(value).astype(float)) > 0 else "silence" for i, value in enumerate(dict["speak_or_not"])]
                one_hot_tensor_list["gender"] = torch.stack([label_to_one_hot(label, "gender") for label in updated_gender])
            else:
                one_hot_tensor_list[type] = torch.stack([label_to_one_hot(label, type) for label in dict[type]])

        keys = dict["keys"]
        details_time = torch.as_tensor(np.array(dict["details_time"]))

        if(one_file):
            behaviour = None
        else:
            behaviour = torch.as_tensor(np.array(dict["behaviour"]))
        
        constants.seq_len = speak_or_not.shape[1]
        constants.audio_dim = audio.shape[2]
        
        return audio, behaviour, speak_or_not, details_time, keys, one_hot_tensor_list, dict["final_behaviour"], dict["final_key"]


    def prepare_data(self):
        if not self.is_prepared:
            print("Lauching of prepare_data")   
            if self.stage == "fit":
                self.X_train_audio, self.Y_train, self.speak_or_not_train, self.details_time_train, _, self.one_hot_tensor_list_train, _, _ = self.load_data("train1")
                self.X_dev_audio, self.Y_dev, self.speak_or_not_dev, self.details_time_dev, self.keys_dev, self.one_hot_tensor_list_dev, _, _ = self.load_data("val")
            
            elif self.stage == "predict":
                if self.file != None:
                    self.X_test_audio, _, self.speak_or_not_test, self.details_time_test, self.keys_test, self.one_hot_tensor_list_test, _, _ = self.load_one_file(self.file)

                elif self.set == "val" or self.set == "test":
                    self.X_test_audio, self.Y_test, self.speak_or_not_test, self.details_time_test, self.keys_test, self.one_hot_tensor_list_test, _, _ = self.load_data(self.set)
                
                else:
                    raise ValueError("CustomDataset - prepare_data - predict - The set is not val or test or a file")
            
            else:
                raise ValueError("CustomDataset - prepare_data - The stage is not fit or predict")



    def setup(self, stage=None):
        print("Lauching of setup - ", self.stage)
        if not self.is_prepared:   
            dir_scaler = join(constants.saved_path, "scaler")
            os.makedirs(dir_scaler, exist_ok=True)

            if stage == 'fit':
                self.X_train_audio_scaled, self.x_scaler = scale_from_scratch(self.X_train_audio, "tanh")
                pickle.dump(self.x_scaler, open(join(dir_scaler, 'scaler_x.pkl'), 'wb'))
                self.Y_scaled_train, self.y_scaler = scale_from_scratch(self.Y_train, "tanh")
                pickle.dump(self.y_scaler, open(join(dir_scaler, 'scaler_y.pkl'), 'wb'))
                self.train_dataset = CustomDataset(X_audio=self.X_train_audio_scaled, Y=self.Y_scaled_train, speak_or_not=self.speak_or_not_train, labels=self.one_hot_tensor_list_train)

                self.X_dev_audio_scaled = scale(self.X_dev_audio, self.x_scaler)
                self.Y_scaled_dev = scale(self.Y_dev, self.y_scaler)
                self.dev_dataset = CustomDataset(X_audio=self.X_dev_audio_scaled, Y=self.Y_scaled_dev, speak_or_not=self.speak_or_not_dev, labels=self.one_hot_tensor_list_dev)

                if(len(constants.designed_targets) > 0):
                    self.designed_dataset = DesignedDataset1(self.X_train_audio_scaled, self.Y_scaled_train, self.speak_or_not_train, self.one_hot_tensor_list_train) #fake designed exemples


            elif stage == "predict":
                self.x_scaler = pickle.load(open(join(dir_scaler, 'scaler_x.pkl'), 'rb'))
                self.y_scaler = pickle.load(open(join(dir_scaler, 'scaler_y.pkl'), 'rb'))
                self.X_test_audio_scaled = scale(self.X_test_audio, self.x_scaler)
                self.test_dataset = CustomDataset(X_audio=self.X_test_audio_scaled, speak_or_not=self.speak_or_not_test, labels=self.one_hot_tensor_list_test, details_time=self.details_time_test, keys=self.keys_test, predict=True)

            
            self.is_prepared = True

        
    def train_dataloader(self):
        print("train dataloader")
        real_loader = DataLoader(self.train_dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=True, 
                                 num_workers=10, 
                                 pin_memory=True,
                                 persistent_workers=True)
        print("number of sequences in train", len(self.train_dataset))
        if(len(constants.designed_targets) > 0):
            designed_loader = DataLoader(self.designed_dataset, 
                                         batch_size=self.batch_size, 
                                         shuffle=True, 
                                         num_workers=10, 
                                         pin_memory=True, 
                                         persistent_workers=True)
            
            return {"real": real_loader, "designed": designed_loader}

    def val_dataloader(self):
        print("val dataloader")
        print("number of sequences in val", len(self.dev_dataset))
        return DataLoader(self.dev_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=False, 
                            num_workers=10, 
                            pin_memory=True, 
                            persistent_workers=True)

    def predict_dataloader(self):
        print("predict dataloader")
        print("number of sequences in test", len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

