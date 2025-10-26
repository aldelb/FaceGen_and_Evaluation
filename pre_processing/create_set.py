import argparse
from math import ceil
import os
from os.path import join, isfile
from os import listdir
import sys
import time
import pandas as pd
import numpy as np
import pickle
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel


CLUSTER="jean-zay"
OPENFACE_FEATURES = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
OPENSMILE_FEATURES = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", 
                      "mfcc3_sma3", "mfcc4_sma3", "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", 
                      "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz", "F2bandwidth_sma3nz", 
                      "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz"]


def change_timestep(df, objective_step, begin_column="begin", end_column="end"):
    current_time = 0
    last_index = len(df)-1
    new_df = pd.DataFrame([], columns=df.columns)
    index = 0
    while(index <= last_index):
        while(current_time + objective_step <= df.at[index, end_column]):
            new_row = df.loc[index].copy()
            new_row['timestamp'] = current_time
            new_df = pd.concat([new_df, new_row.to_frame().T], ignore_index=True)
            current_time = round(current_time + objective_step,2)
        index = index + 1
    return new_df

def putListeningToZero(df, features):
    new_df = df.copy()
    index_listening_behavior = df.where(df['bool_speak'] == 0).dropna().index
    new_df.loc[index_listening_behavior, features] = 0
    return new_df

def create_result_df(key, path_dict, timestep, regenerate_flag):
    df_ipu = pd.read_excel(path_dict["ipu_path"]+key+".xlsx")[["begin", "end", "speak", "bool_speak", "dialog_act", "valence", "arousal", "certainty", "dominance"]]

    # Check if alignment file exists or if regeneration is needed
    df_ipu_align_path = join(path_dict["ipu_path"], "align", key+".csv")
    if(not isfile(df_ipu_align_path) or regenerate_flag):
        # Align the IPU dataframe with the given timestep and save it as a CSV
        df_ipu_align = change_timestep(df_ipu, timestep)
        df_ipu_align.to_csv(df_ipu_align_path, index=False)
    else:
        df_ipu_align = pd.read_csv(df_ipu_align_path)
    # Ensure that the 'timestamp' column is of type float
    df_ipu_align.timestamp = df_ipu_align.timestamp.astype(float)

    df_video = pd.read_csv(path_dict["openface_path"]+key+".csv")[["timestamp"] + OPENFACE_FEATURES]
    df_audio = pd.read_csv(path_dict["opensmile_path"]+key+".csv")[["timestamp"] + OPENSMILE_FEATURES]

    # Define the path to save or load the merged result
    df_result_path = join(path_dict["init_output_path"], key+".csv")
    # Check if the result file exists or needs regeneration
    if(not isfile(df_result_path) or regenerate_flag):
        # Merge video data with IPU alignment on 'timestamp', drop unnecessary columns
        df_result = df_video.merge(df_ipu_align, on='timestamp', how='inner')
        df_result = df_result.merge(df_audio, on='timestamp', how='inner')
        df_result.drop(["begin", "end"], axis=1, inplace=True)
        df_result.to_csv(df_result_path, index=False)
    else:
        df_result = pd.read_csv(df_result_path)
    
    df_result = putListeningToZero(df_result, ["AU25_r", "AU26_r"]) 
    #we choose to put 0 to the AU25 and AU26 when the person is listening
    #for the speaker A and B
    
    df_result_listening_to_zero = putListeningToZero(df_result, OPENFACE_FEATURES)
    #other possibilities : all behaviour to 0 when listening


    

    # Get the last timestamp from the merged result
    end_time_result = df_result["timestamp"].iloc[-1]

    return df_result, df_result_listening_to_zero, end_time_result
    

def create_wav2vec_embedding(wav_path, t1, t2, processor):
        #number_of_frame = ceil((t2-t1)*fps)
        t2 = t2 + 0.1 #to get the last frame
        speech_array, _ = librosa.load(wav_path, offset=t1, duration=t2-t1, sr=16000)
        input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values

        # #TODO : remove, its done in the model
        # audio_encoder = HubertModel.from_pretrained("/lustre/fswork/projects/rech/urk/uln35en/model/hubert-large-ls960-ft/")
        # audio_encoder.feature_extractor._freeze_parameters()
        # for _, param in audio_encoder.named_parameters():
        #     param.requires_grad = False
        # embedding = audio_encoder(input_values, output_hidden_states=True)
        # print(embedding.last_hidden_state.shape)

        return input_values

    
def create_set(dataset_name, regenerate_flag, segment_length, overlap, timestep):

    dic_paths = getPath(dataset_name, segment_length)
    details_df = pd.read_excel(dic_paths["details_file"])
    data_details = details_df.set_index("nom").to_dict(orient='index')
    
    print("**length", segment_length, "overlap:", overlap, "nombre de frames:", int(segment_length/timestep))

    processor = Wav2Vec2Processor.from_pretrained(dic_paths["modelname"])

    for file in listdir(dic_paths["openface_path"]):
        if(".csv" not in file):
            continue
        key_A = file[0:-4]
        if("mic1" in key_A):
            key_B = key_A.replace("mic1", "mic2")
        else:
            key_B = key_A.replace("mic2", "mic1")

        if(key_A not in data_details.keys() or not isfile(join(dic_paths["wav_path"], key_A + ".wav"))):
            continue 

        print("*"*10, "Process of", key_A)

        final_dict = {"key": key_A, "key_speakerB": key_B,
                        "time_array": [], "previous_time_array": [], "details_time": [],
                        "wav2vec": [], "previous_wav2vec":[], 
                        "behaviour": [], "previous_behaviour":[], 
                        "behaviour_listening_to_zero": [], "previous_behaviour_listening_to_zero": [],
                        "audio": [], "previous_audio": [],
                        "speak_or_not": [], "previous_speak_or_not":[],
                        "wav2vec_speakerB": [], "audio_speakerB": [], "behaviour_speakerB": [], "speak_or_not_speakerB": []}
        
        for label in ["dialog_act", "valence", "dominance", "certainty", "arousal"]:
            final_dict[label] = []
            final_dict[label+"_speakerB"] = []
        
        final_path  = join(dic_paths["output_path"],  key_A+".p")
            
        df_result, df_result_listening_to_zero, end_time_result = create_result_df(key_A, dic_paths, timestep, regenerate_flag)
        final_dict["final_behaviour"] = df_result[OPENFACE_FEATURES]
        final_dict["final_behaviour_listening_to_zero"] = df_result_listening_to_zero[OPENFACE_FEATURES]
        df_result_speakerB, _, end_time_result_speakerB = create_result_df(key_B, dic_paths, timestep, regenerate_flag)
        final_dict["final_behaviour_speakerB"] = df_result_speakerB[OPENFACE_FEATURES]

        # Store additional speaker information from data_details
        final_dict["gender"] = data_details[key_A]["genre"]
        final_dict["role"] = data_details[key_A]["role"]
        final_dict["attitude"] = data_details[key_A]["attitude"]

        # Store additional speaker B information from data_details
        final_dict["gender_speakerB"] = data_details[key_B]["genre"]
        final_dict["role_speakerB"] = data_details[key_B]["role"]
        final_dict["attitude_speakerB"] = data_details[key_B]["attitude"]

        #cut into segment of length "segment_length" with overlap, and create the array of features
        t1, t2 = 0, segment_length

        #create the previous for the first segments 
        silence_wav2vec = create_wav2vec_embedding(dic_paths["silence_wav_path"], 0, segment_length, processor)
        with open(join(dic_paths["output_path"],"silence_wav2vec.p"), 'wb') as f:
            pickle.dump(silence_wav2vec, f)
        zero_df = pd.DataFrame(0, index=range(int(segment_length/timestep)), columns=df_result.columns)

        previous_wav2vec = silence_wav2vec
        previous_behaviour = zero_df[OPENFACE_FEATURES].values
        previous_behaviour_listening_to_zero = zero_df[OPENFACE_FEATURES].values
        previous_audio = zero_df[OPENSMILE_FEATURES].values
        previous_time_array = [-segment_length,0]
        previous_speak_or_not = zero_df["bool_speak"].values
        
        while t2 <= end_time_result and t2 <= end_time_result_speakerB:
            print("Times:", t1, "|", t2)
            print("Previous times", previous_time_array[0], "|", previous_time_array[1])

            cut = df_result[(df_result["timestamp"] < t2) & (df_result["timestamp"] >= t1)]
            cut_listening_to_zero = df_result_listening_to_zero[(df_result_listening_to_zero["timestamp"] < t2) & (df_result_listening_to_zero["timestamp"] >= t1)]
            cut_speakerB = df_result_speakerB[(df_result_speakerB["timestamp"] < t2) & (df_result_speakerB["timestamp"] >= t1)]

            final_dict["previous_time_array"].append(previous_time_array)
            final_dict["previous_wav2vec"].append(previous_wav2vec)
            final_dict["previous_audio"].append(previous_audio)
            final_dict["previous_behaviour"].append(previous_behaviour)
            final_dict["previous_behaviour_listening_to_zero"].append(previous_behaviour_listening_to_zero)
            final_dict["previous_speak_or_not"].append(previous_speak_or_not)
            
            #time
            previous_time_array = [t1, t2]
            final_dict["time_array"].append(previous_time_array)
            final_dict["details_time"].append(cut["timestamp"].values)

            #hubert
            previous_wav2vec = create_wav2vec_embedding(dic_paths["wav_path"]+final_dict["key"]+".wav", t1, t2, processor)
            final_dict["wav2vec"].append(previous_wav2vec)
            final_dict["wav2vec_speakerB"].append(create_wav2vec_embedding(dic_paths["wav_path"]+final_dict["key_speakerB"]+".wav", t1, t2, processor))
            
            ## behaviour : openface features
            previous_behaviour = cut[OPENFACE_FEATURES].values
            previous_behaviour_listening_to_zero = cut_listening_to_zero[OPENFACE_FEATURES].values
            final_dict["behaviour"].append(previous_behaviour)
            final_dict["behaviour_listening_to_zero"].append(previous_behaviour_listening_to_zero)
            final_dict["behaviour_speakerB"].append(cut_speakerB[OPENFACE_FEATURES].values)

            ## audio : opensmile features
            previous_audio = cut[OPENSMILE_FEATURES].values
            final_dict["audio"].append(previous_audio)
            final_dict["audio_speakerB"].append(cut_speakerB[OPENSMILE_FEATURES].values)

            # List of label keys for both speakers 
            for label in ["dialog_act", "valence", "dominance", "certainty", "arousal"]:
                final_dict[label].append(cut[label].values)
                final_dict[label+"_speakerB"].append(cut_speakerB[label].values)

            # Speak or not
            previous_speak_or_not = cut["bool_speak"].values
            final_dict["speak_or_not"].append(previous_speak_or_not)
            final_dict["speak_or_not_speakerB"].append(cut_speakerB["bool_speak"].values)

            t1, t2 = round(t1 + segment_length - overlap,2), round(t2 + segment_length - overlap,2)

        with open(final_path, 'wb') as f:
            pickle.dump(final_dict, f)
            print("")
        del final_dict


def getPath(dataset_name, segment_length):
    dic_path = {}
    if(CLUSTER=="jean-zay"):
        general_path = "/lustre/fswork/projects/rech/urk/uln35en/"

        dic_path["silence_wav_path"] = general_path + "/Projets/non-verbal-behaviours-generation/pre_processing/silence/silence.wav"
        dic_path["modelname"] = general_path + "model/hubert-large-ls960-ft/"
        # french_modelname = general_path + "model/exp_w2v2t_fr_hubert_s767/"

        dataset_path = general_path + "raw_data/"+dataset_name+"/"
        init_output_path = dataset_path + "/final_data/"
        output_path = join(init_output_path, str(segment_length))
        os.makedirs(output_path, exist_ok=True)
        
        dic_path["init_output_path"] = init_output_path
        dic_path["output_path"] = output_path
        dic_path["wav_path"] = dataset_path + "audio/full/"
        dic_path["openface_path"] = dataset_path + "video/processed/" 
        dic_path["opensmile_path"] = dataset_path + "audio/processed/"
        dic_path["ipu_path"] = dataset_path + "annotation/processed/ipu_with_tag/"
        dic_path["details_file"] = dataset_path + "details.xlsx"

    else:
        sys.exit("Error in the cluster name")
    return dic_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset')
    parser.add_argument('-regenerate', action='store_true')
    parser.add_argument('-segment', type=int, default=4)
    parser.add_argument('-overlap', type=int, default=0.4)
    parser.add_argument('-timestep', type=float, default=0.04)
    args = parser.parse_args()

    dataset_name = args.dataset
    segment_length = args.segment #secondes
    overlap_value = args.overlap #secondes 
    timestep = args.timestep
    regenerate_flag = args.regenerate
    #hidden_layer = 12 #None for all other layers 

    print("*"*10, "regenerate flag:", regenerate_flag)

    begin_time = time.time()
    create_set(dataset_name, regenerate_flag, segment_length, overlap_value, timestep)
    print("*"*10, "end of creation", "*"*10)

    end_time = time.time()
    print(f"Total time taken: {end_time - begin_time} seconds")

    return 0

if __name__ == "__main__":
    
    sys.exit(main())
