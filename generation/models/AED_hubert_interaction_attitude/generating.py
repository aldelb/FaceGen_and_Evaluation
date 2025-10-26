import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import utils.constants.constants as constants
import utils.constants.features as features
from utils.data.data_utils import reshape_output
from utils.data.labels import categories, label_to_one_hot
import torch

class GenerateModel():
    def __init__(self):
        super(GenerateModel, self).__init__()


    def generate_motion(self, model, trainer_args, dm):
        dataloader = dm.predict_dataloader()
        model.eval()

        all_files = {}
        prev_behav_item = {}
        for key in categories["small_attitude"]:
            prev_behav_item[key] = torch.zeros((1, constants.seq_len, constants.pose_size + constants.au_size)).to(model.device)

        for data in dataloader:
            prev_audio, audio, audio_speakerB, behav_speakerB, details_time, keys_list, attitude_label = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

            for i, key_item in enumerate(keys_list):
                details_time_item = details_time[i]
                # print("Part", "[", details_time_item[0].item(), ",", details_time_item[-1].item(), "] of", key_item)
                prev_audio_item = prev_audio[i]
                audio_item = audio[i]
                audio_speakerB_item = audio_speakerB[i]
                behav_speakerB_item = behav_speakerB[i]
                # attitude_item = attitude_label[i] #one_hot vector

                if(key_item not in all_files.keys()): #process of a new video
                    # print("New video:", key_item)
                    for attitude_name in categories["small_attitude"]:
                        if attitude_name == "None":
                            all_files[key_item] = []
                        else:
                            all_files[key_item + "_" + attitude_name] = []
                        prev_behav_item[attitude_name] = torch.zeros((1, constants.seq_len, constants.pose_size + constants.au_size)).to(audio_item.device)



                for attitude_name in categories["small_attitude"]:
                    print("Attitude:", attitude_name)
                    attitude_item = label_to_one_hot(attitude_name, "small_attitude").to(audio_item)
                    rescaled_current_behav_item, current_behav_item = self.generate_one_sequence(model, attitude_item, prev_audio_item, audio_item, prev_behav_item[attitude_name], audio_speakerB_item, behav_speakerB_item)
                    out = np.concatenate((np.array(details_time_item).reshape(-1,1), rescaled_current_behav_item[:,:constants.eye_size], np.zeros((rescaled_current_behav_item.shape[0], 3)), rescaled_current_behav_item[:,constants.eye_size:]), axis=1)
                    df = pd.DataFrame(data = out, columns = features.OUTPUT_COLUMNS)
                    if attitude_name == "None":
                        all_files[key_item].append(df)
                    else:
                        all_files[key_item + "_" + attitude_name].append(df)

                    prev_behav_item[attitude_name] = current_behav_item

        return all_files
    
    def generate_one_sequence(self, model, attitude, prev_audio, audio, prev_behav, audio_speakerB, behav_speakerB):
        with torch.no_grad():
            _, output_eye, output_pose_r, output_au = model(attitude.unsqueeze(0), prev_audio.unsqueeze(0), audio.unsqueeze(0), prev_behav, audio_speakerB.unsqueeze(0), behav_speakerB.unsqueeze(0))

        raw_pred = torch.cat((output_eye, output_pose_r, output_au), 2)
        rescaled_pred = reshape_output(output_eye, output_pose_r, output_au, model.pose_scaler).squeeze(0)

        return rescaled_pred, raw_pred
    
    def generate_latent(self, model, trainer_args, dm):
        #TODO: Implement this method if needed
        pass