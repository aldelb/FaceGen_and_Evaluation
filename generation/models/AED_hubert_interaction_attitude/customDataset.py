import pandas as pd
import pytorch_lightning as pl
import torch
import os
from os.path import join, isdir, isfile
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset

import utils.constants.features as features
import utils.constants.constants as constants
from utils.data.data_utils import scale_from_scratch, scale
from utils.data.labels import label_to_one_hot, get_maj_label, categories
from utils.data.designedDataset import DesignedDataset2
from utils.data.hubert_embedding import HubertEmbedding



class CustomDataset(Dataset):
    def __init__(self, keys=None, prev_audio=None, audio=None, prev_behav=None, behav=None, audio_speakerB=None, behav_speakerB=None, speak_or_not=None, labels=None, details_time=None, final_Y=None, predict=False, final_keys=None):
        self.prev_audio = prev_audio
        self.audio = audio
        self.prev_behav = prev_behav
        self.behav = behav
        self.audio_speakerB = audio_speakerB
        self.behav_speakerB = behav_speakerB
        self.speak_or_not = speak_or_not
        self.labels = labels
        self.details_time = details_time
        self.final_Y = final_Y
        self.predict = predict
        self.keys = keys
        self.final_keys = final_keys

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, i):
        if(self.predict):
            base_output = self.prev_audio[i], self.audio[i], self.audio_speakerB[i], self.behav_speakerB[i], self.details_time[i], self.keys[i], self.labels["small_attitude"][i]
            return base_output
        
        else:
            base_output = self.prev_audio[i], self.audio[i], self.prev_behav[i], self.behav[i], self.audio_speakerB[i], self.behav_speakerB[i], self.labels["small_attitude"][i]
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
            "prev_audio":[], "audio": [], "prev_behaviour": [], "behaviour": [],
            "behaviour_speakerB": [], "audio_speakerB": [], 
            "prev_details_time": [], "details_time": [], 
            "prev_speak_or_not": [], "speak_or_not": [], "speak_or_not_speakerB": [],
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
        
        final_dict["audio"].extend(hubert_embedding)

        if "hubert_speakerB" not in data.keys():
            data["hubert_speakerB"] = {}

        if constants.hidden_state_index not in data["hubert_speakerB"].keys():
            print("The file does not contain hubert embedding, creation...")
            audio_data = torch.stack(data["wav2vec_speakerB"], 0)
            hubert_embedding_speakerB = hubert_embedding_extractor.create_hubert_embedding(audio_data.squeeze(1))
            data["hubert_speakerB"][constants.hidden_state_index] = hubert_embedding_speakerB
            pickle.dump(data, open(path_file, 'wb'))
            print("The file has been updated with hubert embedding")
        else:
            hubert_embedding_speakerB = data["hubert_speakerB"][constants.hidden_state_index]
        
        final_dict["audio_speakerB"].extend(hubert_embedding_speakerB)


        #previous_wav2vec
        if "hubert_prev" not in data.keys():
            data["hubert_prev"] = {}
        
        if constants.hidden_state_index not in data["hubert_prev"].keys():
            print("The file does not contain previous hubert embedding, creation...")
            audio_data = torch.stack(data["previous_wav2vec"], 0)
            hubert_embedding_prev = hubert_embedding_extractor.create_hubert_embedding(audio_data.squeeze(1))
            data["hubert_prev"][constants.hidden_state_index] = hubert_embedding_prev
            pickle.dump(data, open(path_file, 'wb'))
            print("The file has been updated with previous hubert embedding")
        else:
            hubert_embedding_prev = data["hubert_prev"][constants.hidden_state_index]

        final_dict["prev_audio"].extend(hubert_embedding_prev)

        
        final_dict["keys"].extend([ele for ele in [data["key"]] for _ in range(len(data["speak_or_not"]))])
        final_dict["details_time"].extend(data["details_time"])
        final_dict["speak_or_not"].extend(data["speak_or_not"])
        final_dict["prev_speak_or_not"].extend(data["previous_speak_or_not"])
        final_dict["speak_or_not_speakerB"].extend(data["speak_or_not_speakerB"])


        for type in constants.main_list_for_loading:
            if(type in ["gender", "attitude", "role"]):
                final_dict[type].extend([ele for ele in [data[type]] for _ in range(len(data["speak_or_not"]))])
            else:
                final_dict[type].extend([get_maj_label(ele) for ele in data[type]])
        
        final_dict["prev_behaviour"].extend(data["previous_behaviour"])
        final_dict["behaviour"].extend(data["behaviour"])
        final_dict["behaviour_speakerB"].extend(data["behaviour_speakerB"])

        if(one_file):
            final_dict["final_behaviour"].append(None)
        else:
            final_dict["final_behaviour"].append(data["final_behaviour"])

        final_dict["final_key"].append(data["key"])

        del data # Free up memory occupied by the 'data' variable

        return final_dict
    

    def update_attitudes(self, behav, attitudes):
        for seq_index in range(behav.shape[0]):
            current_attitude = attitudes[seq_index]
            if(current_attitude not in categories["small_attitude"]): #only keep the sequences with "colere_chaude" or "conciliant"
                attitudes[seq_index] = "None"
                continue

            others_au_tensor = behav[seq_index,:, features.INDEX_AUS_WT_SPEAK]
            
            #criteria of selection of "colere_chaude"
            if(current_attitude == "Colere_chaude"):
                au = "AU04_r"
            #criteria of selection of "conciliant"
            elif(current_attitude == "Conciliant"):
                au = "AU12_r"
            au_index = features.OPENFACE_FEATURES.index(au)  
            au_tensor = behav[seq_index,:,au_index] 

            if (((au_tensor >= 1.5)).sum().item()/au_tensor.shape[0] > 0.1): #if more than X% of the values are greater than 1.5
                attitudes[seq_index] = current_attitude
            elif ((others_au_tensor < 1.5).sum().item()/(others_au_tensor.shape[0]*others_au_tensor.shape[1]) > 0.95):
                #print((others_au_tensor <= 1.5).sum().item())
                attitudes[seq_index] = "Neutral"
            else:
                attitudes[seq_index] = "None"

        return attitudes


    def format_dict(self, dict, one_file=False):

        speak_or_not = torch.as_tensor(np.array(dict["speak_or_not"], dtype=np.float32))
        speak_or_not_extended = torch.repeat_interleave(speak_or_not, 2, dim=1).unsqueeze(-1)
        audio = torch.stack(dict["audio"] , 0).squeeze()
        audio = torch.cat((speak_or_not_extended, audio), dim=2)

        speak_or_not_speakerB = torch.as_tensor(np.array(dict["speak_or_not_speakerB"], dtype=np.float32))
        speak_or_not_speakerB_extended = torch.repeat_interleave(speak_or_not_speakerB, 2, dim=1).unsqueeze(-1)
        audio_speakerB = torch.stack(dict["audio_speakerB"] , 0).squeeze()
        audio_speakerB = torch.cat((speak_or_not_speakerB_extended, audio_speakerB), dim=2)

        prev_speak_or_not = torch.as_tensor(np.array(dict["prev_speak_or_not"], dtype=np.float32))
        prev_speak_or_not_extended = torch.repeat_interleave(prev_speak_or_not, 2, dim=1).unsqueeze(-1)
        prev_audio = torch.stack(dict["prev_audio"] , 0).squeeze()
        prev_audio = torch.cat((prev_speak_or_not_extended, prev_audio), dim=2)

        prev_behaviour = torch.as_tensor(np.array(dict["prev_behaviour"]))
        behaviour = torch.as_tensor(np.array(dict["behaviour"]))
        behaviour_speakerB = torch.as_tensor(np.array(dict["behaviour_speakerB"]))

        one_hot_tensor_list = {}
        for type in constants.main_list_for_loading:
            if(type in ["gender"]):
                updated_gender = [dict["gender"][i] if np.mean(np.array(value).astype(float)) > 0 else "silence" for i, value in enumerate(dict["speak_or_not"])]
                one_hot_tensor_list["gender"] = torch.stack([label_to_one_hot(label, "gender") for label in updated_gender])
            else:
                one_hot_tensor_list[type] = torch.stack([label_to_one_hot(label, type) for label in dict[type]])

        if(one_file):
            dict["small_attitude"] = ["None" for _ in range(len(dict["speak_or_not"]))]
            one_hot_tensor_list["small_attitude"] = torch.stack([label_to_one_hot(label, "small_attitude") for label in dict["small_attitude"]])

        else:
            dict["small_attitude"] = self.update_attitudes(behaviour, dict["attitude"])
            one_hot_tensor_list["small_attitude"] = torch.stack([label_to_one_hot(label, "small_attitude") for label in dict["small_attitude"]])
        

        keys = dict["keys"]
        details_time = torch.as_tensor(np.array(dict["details_time"]))
        
        constants.seq_len = speak_or_not.shape[1]
        constants.audio_dim = audio.shape[2]
        
        return prev_audio, audio, prev_behaviour, behaviour, speak_or_not, audio_speakerB, behaviour_speakerB, details_time, keys, one_hot_tensor_list, dict["final_behaviour"], dict["final_key"]


    def prepare_data(self):
        if not self.is_prepared:
            print("Lauching of prepare_data")   
            if self.stage == "fit":
                self.prev_audio_train, self.audio_train, self.prev_behaviour_train, self.behaviour_train, self.speak_or_not_train, self.audio_speakerB_train, self.behaviour_speakerB_train, self.details_time_train, self.keys_train, self.one_hot_tensor_list_train, _, _ = self.load_data("train1")
                self.prev_audio_dev, self.audio_dev, self.prev_behaviour_dev, self.behaviour_dev, self.speak_or_not_dev, self.audio_speakerB_dev, self.behaviour_speakerB_dev, self.details_time_dev, self.keys_dev, self.one_hot_tensor_list_dev, _, _ = self.load_data("val")
            
            elif self.stage == "predict":
                if self.file != None:
                    self.prev_audio_test, self.audio_test, self.prev_behaviour_test, self.behaviour_test, self.speak_or_not_test, self.audio_speakerB_test, self.behaviour_speakerB_test, self.details_time_test, self.keys_test, self.one_hot_tensor_list_test, _, _ = self.load_one_file(self.file)

                elif self.set == "val" or self.set == "test":
                    self.prev_audio_test, self.audio_test, self.prev_behaviour_test, self.behaviour_test, self.speak_or_not_test, self.audio_speakerB_test, self.behaviour_speakerB_test, self.details_time_test, self.keys_test, self.one_hot_tensor_list_test, _, _ = self.load_data(self.set)
                
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
                self.audio_scaled_train, self.x_scaler = scale_from_scratch(self.audio_train, "tanh")
                self.prev_audio_scaled_train = scale(self.prev_audio_train, self.x_scaler)
                self.audio_speakerB_scaled_train = scale(self.audio_speakerB_train, self.x_scaler)
                pickle.dump(self.x_scaler, open(join(dir_scaler, 'scaler_x.pkl'), 'wb'))


                self.behav_scaled_train, self.y_scaler = scale_from_scratch(self.behaviour_train, "tanh")
                self.prev_behav_scaled_train = scale(self.prev_behaviour_train, self.y_scaler)
                self.behav_speakerB_scaled_train = scale(self.behaviour_speakerB_train, self.y_scaler)
                pickle.dump(self.y_scaler, open(join(dir_scaler, 'scaler_y.pkl'), 'wb'))

                self.train_dataset = CustomDataset(prev_audio=self.prev_audio_scaled_train, audio=self.audio_scaled_train, prev_behav=self.prev_behav_scaled_train, behav=self.behav_scaled_train, audio_speakerB=self.audio_speakerB_scaled_train, behav_speakerB=self.behav_speakerB_scaled_train,speak_or_not=self.speak_or_not_train, labels=self.one_hot_tensor_list_train)

                self.prev_audio_scaled_dev = scale(self.prev_audio_dev, self.x_scaler)
                self.audio_scaled_dev = scale(self.audio_dev, self.x_scaler)
                self.audio_speakerB_scaled_dev = scale(self.audio_speakerB_dev, self.x_scaler)
                self.behav_scaled_dev = scale(self.behaviour_dev, self.y_scaler)
                self.prev_behav_scaled_dev = scale(self.prev_behaviour_dev, self.y_scaler)
                self.behav_speakerB_scaled_dev = scale(self.behaviour_speakerB_dev, self.y_scaler)
                self.dev_dataset = CustomDataset(prev_audio=self.prev_audio_scaled_dev, audio=self.audio_scaled_dev, prev_behav=self.prev_behav_scaled_dev, behav=self.behav_scaled_dev, audio_speakerB=self.audio_speakerB_scaled_dev, behav_speakerB=self.behav_speakerB_scaled_dev, speak_or_not=self.speak_or_not_dev, labels=self.one_hot_tensor_list_dev)

                if(len(constants.designed_targets) > 0):
                    self.designed_dataset = DesignedDataset2(self.prev_audio_scaled_train,
                                            self.audio_scaled_train,
                                            self.behav_scaled_train, 
                                            self.prev_behav_scaled_train, 
                                            self.speak_or_not_train, 
                                            self.audio_speakerB_scaled_train, 
                                            self.behav_speakerB_scaled_train, 
                                            self.one_hot_tensor_list_train)


            elif stage == "predict":
                self.x_scaler = pickle.load(open(join(dir_scaler, 'scaler_x.pkl'), 'rb'))
                self.y_scaler = pickle.load(open(join(dir_scaler, 'scaler_y.pkl'), 'rb'))
                self.prev_audio_scaled_test = scale(self.prev_audio_test, self.x_scaler)
                self.audio_scaled_test = scale(self.audio_test, self.x_scaler)
                self.audio_speakerB_scaled_test = scale(self.audio_speakerB_test, self.x_scaler)
                self.behav_speakerB_scaled_test = scale(self.behaviour_speakerB_test, self.y_scaler)
                self.test_dataset = CustomDataset(prev_audio=self.prev_audio_scaled_test,
                                            audio=self.audio_scaled_test, 
                                            audio_speakerB=self.audio_speakerB_scaled_test,
                                            behav_speakerB=self.behav_speakerB_scaled_test,
                                            speak_or_not=self.speak_or_not_test, 
                                            labels=self.one_hot_tensor_list_test, 
                                            details_time=self.details_time_test, 
                                            keys=self.keys_test, 
                                            predict=True,
                                            final_keys=self.keys_test)

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
    
    def predict_dataloader_shuffle(self):
        print("predict dataloader")
        print("number of sequences in test", len(self.test_dataset))
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
