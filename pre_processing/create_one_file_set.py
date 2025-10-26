from os import listdir
from os.path import isfile, join
import opensmile
import speechpy
import numpy as np
import pandas as pd
import pickle
import sys
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel
import argparse

visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


def getPath():
    path = "/gpfsdswork/projects/rech/urk/uln35en/raw_data/audio_file/"
    wav_dir = "wav/"
    processed_dir = "processed/"
    anno_dir = "annotation/"
    complete_anno_dir = "complete_annotation/"
    set_dir = "final_data/"

    return path, wav_dir, processed_dir, anno_dir, complete_anno_dir, set_dir


def createCsvFile(wav_file, origin_processed_file):
    print("*"*10, "createCsvFile", "*"*10)
    smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
    result_smile = smile.process_file(wav_file)
    result_smile.to_csv(origin_processed_file, sep=',')

def removeOverlapAndAlignStep(origin_processed_file, processed_file, objective_step):
    print("*"*10, "removeOverlapAndAlignStep", "*"*10)
    df_audio = pd.read_csv(origin_processed_file)
    df_audio["start"] = df_audio["start"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    df_audio["end"] = df_audio["end"].transform(lambda x :  pd.Timedelta(x).total_seconds())
    #we remove the overlaps and recalculate with averages
    df_audio_wt_overlap = df_audio.copy()
    df_audio_wt_overlap = df_audio_wt_overlap.rename(columns={"start":"timestamp"})
    df_audio_wt_overlap = df_audio_wt_overlap.drop(columns=["end"])
    df_audio_wt_overlap = df_audio_wt_overlap[audio_features]
    #mean the value of overlap
    for index, row in df_audio_wt_overlap.iterrows():
        if index != 0:
            df_audio_wt_overlap.at[index, "Loudness_sma3"] = (row["Loudness_sma3"] + df_audio.iloc[[index-1]]["Loudness_sma3"]) / 2

    #we change the timestep to match the openface timestep
    df_audio_wt_overlap["timestamp"] = df_audio_wt_overlap["timestamp"].astype(float)
    df_audio_wt_overlap["timestamp"] = ((df_audio_wt_overlap["timestamp"]/objective_step).astype(int)) * objective_step
    df_audio_wt_overlap["timestamp"] = round(df_audio_wt_overlap["timestamp"],2)
    df_audio_wt_overlap = df_audio_wt_overlap.groupby(by=["timestamp"]).mean()
    df_audio_wt_overlap = df_audio_wt_overlap.reset_index()
    df_audio = df_audio_wt_overlap
    df_audio.to_csv(processed_file, index=False)

def create_complete_ipu_file(anno_file, dialogAct, valence, arousal, certainty, dominance):
    df = pd.read_csv(anno_file, names=["ipu", "begin", "end", "speak"])
    df["bool_speak"] = df["speak"].transform(lambda x : 0 if x == "#" else 1)
    df["dialogAct"] = dialogAct
    df["valence"] = valence
    df["arousal"] = arousal
    df["certainty"] = certainty
    df["dominance"] = dominance
    df = df.drop(columns=['ipu'])
    return df

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

def putListeningToConstant(df, visual_features, listening_value="constant"):
    new_df = df.copy()
    index_listening_behavior = df.where(df['bool_speak'] == 0).dropna().index

    if(listening_value == "zero"):
        new_df.loc[index_listening_behavior, visual_features] = 0

    elif(listening_value == "constant"):
        # extract the different sequences of listening behavior
        sequences = []
        sequence = []
        for i in range(len(index_listening_behavior)-1):
            if index_listening_behavior[i+1] - index_listening_behavior[i]==1:
                sequence.append(index_listening_behavior[i])
            else:
                sequence.append(index_listening_behavior[i])
                sequences.append(sequence)
                sequence = []
        # for these sequences, put the previous behavior features in df
        for sequence in sequences:
            if sequence[0] == 0:
                precedent_behaviour = 0
            else:
                precedent_behaviour = df[visual_features].iloc[sequence[0]-1]
                precedent_behaviour = [round(precedent_behaviour[i], 5) for i in range(len(precedent_behaviour))]
        
            for i in range(len(sequence)):
                    new_df.loc[sequence[i], visual_features] = precedent_behaviour
    
    return new_df

def create_result_df(key, init_output_path, final_dict, visual_features, timestep, moveToZero, listening_value, regenerate_flag, speakerB=False):
    if(speakerB):
        path_elem = "_speakerB"
    else:
        path_elem = ""

    df_ipu = pd.read_excel(final_dict["ipu_path"]+key+".xlsx")[["begin", "end", "speak", "bool_speak", "dialog_act", "valence", "arousal", "certainty", "dominance"]]

    # Check if alignment file exists or if regeneration is needed
    df_ipu_align_path = join(final_dict["ipu_path"], "align", key+".csv")
    if(not isfile(df_ipu_align_path) or regenerate_flag):
        # Align the IPU dataframe with the given timestep and save it as a CSV
        df_ipu_align = change_timestep(df_ipu, timestep)
        df_ipu_align.to_csv(df_ipu_align_path, index=False)
    else:
        df_ipu_align = pd.read_csv(df_ipu_align_path)
    # Ensure that the 'timestamp' column is of type float
    df_ipu_align.timestamp = df_ipu_align.timestamp.astype(float)
    # Extract the last timestamp (end time of annotations)
    end_time_annotations = df_ipu_align["timestamp"].iloc[-1]

    df_video = pd.read_csv(final_dict["behaviour_path"]+key+".csv")[["timestamp"] + visual_features]
    # Extract the last timestamp from the video data
    end_time_video = df_video["timestamp"].iloc[-1]

    # Define the path to save or load the merged result
    df_result_path = join(init_output_path, key+".csv")
    # Check if the result file exists or needs regeneration
    if(not isfile(df_result_path) or regenerate_flag):
        # Merge video data with IPU alignment on 'timestamp', drop unnecessary columns
        df_result = df_video.merge(df_ipu_align, on='timestamp', how='inner')
        df_result.drop(["begin", "end"], axis=1, inplace=True)
        df_result.to_csv(df_result_path, index=False)
    else:
        df_result = pd.read_csv(df_result_path)
    
    if moveToZero: # If moveToZero flag is set, adjust behavior features to a constant listening value
        df_result = putListeningToConstant(df_result, visual_features, listening_value)

    # Get the last timestamp from the merged result
    end_time_result = df_result["timestamp"].iloc[-1]

    # Get the last timestamp from the audio file
    y, sr = librosa.load(final_dict["wav_path"]+key+".wav", sr=None)
    audio_duration = librosa.get_duration(y=y, sr=sr)

    print("End time"+path_elem, "-", key, ":", "[result:", end_time_result, "]", "[ipu:", end_time_annotations, "]", "[video:", end_time_video, "]", "[audio:", audio_duration, "]")

    return df_result, end_time_result
    

def create_hubert_embedding(wav_path, t1, t2, processor, audio_encoder, hidden_layer, fps=50):
        number_of_frame = int((t2-t1)*fps)
        t2 = t2 + 0.05 #to get the last frame
        speech_array, sampling_rate = librosa.load(wav_path, offset=t1, duration=t2-t1, sr=16000)
        input_values = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values
        embedding = audio_encoder(input_values, output_hidden_states=True)
        correct_embedding = []
        for i in range(len(embedding.hidden_states)):
            if(hidden_layer is not None and i == hidden_layer):
                correct_embedding.append(embedding.hidden_states[i][:,:number_of_frame,:])
                #print(embedding.hidden_states[i][:,0:number_of_frame,:].shape)
            else:
                correct_embedding.append(None)
        return correct_embedding

#pour chaque IPU extrait par sppas (IPU1, IPU2...) je veux ajouter direct les labels choisis à la main (donc pas besoin de faire l'extraction gpt)
#Pour le speaker B, silence et ne bouge pas ? 
def create_set(file_name, regenerate_flag, hidden_layer, moveToZero, listening_value, segment_length, shift_value, timestep):
    init_output_path, output_path, audio_path, visual_path, ipu_path, data_details, modelname, french_modelname, silence_wav_path = getPath(dataset_name, moveToZero, segment_length)
    print("process of", file_name)

    modelname = "/gpfsdswork/projects/rech/urk/uln35en/model/hubert-large-ls960-ft/"
    processor = Wav2Vec2Processor.from_pretrained(modelname)
    audio_encoder = HubertModel.from_pretrained(modelname)
    audio_encoder.feature_extractor._freeze_parameters()
    for _, param in audio_encoder.named_parameters():
        param.requires_grad = False
    
    silence_hubert_array = create_hubert_embedding(silence_wav_path, 0, shift_value, processor, audio_encoder, hidden_layer)
    print("timing of silence", shift_value)
    with open(join(output_path,"silence_hubert_array.p"), 'wb') as f:
        pickle.dump(silence_hubert_array, f)

    final_dict = {"wav_path": audio_path, "behaviour_path": visual_path, "ipu_path": ipu_path,
                    "time_array": [], "details_time": [],
                    "hidden_layer": hidden_layer,
                    "hubert_array": [], "hubert_array_speakerB": [],
                    "behaviour": [], "behaviour_speakerB": [],
                    "speak_or_not": [], "speak_or_not_speakerB": []}
    for label in ["dialog_act", "valence", "dominance", "certainty", "arousal"]:
        final_dict[label] = []
        final_dict[label+"_speakerB"] = []

    df_result, end_time_result = create_result_df(final_dict["key"], init_output_path, final_dict, visual_features, timestep, moveToZero, listening_value, regenerate_flag)
    final_dict["final_behaviour"] = df_result[visual_features]
    df_result_speakerB, end_time_result_speakerB = create_result_df(final_dict["key_speakerB"], init_output_path, final_dict, visual_features, timestep, moveToZero, listening_value, regenerate_flag, speakerB=True)
    final_dict["final_behaviour_speakerB"] = df_result_speakerB[visual_features]

    # Store additional speaker information from data_details
    final_dict["gender"] = data_details[key_A]["genre"]
    final_dict["role"] = data_details[key_A]["role"]
    final_dict["set"] = data_details[key_A]["set"]
    final_dict["attitude"] = data_details[key_A]["attitude_harceleur"]

    # Store additional speaker B information from data_details
    final_dict["gender_speakerB"] = data_details[key_B]["genre"]
    final_dict["role_speakerB"] = data_details[key_B]["role"]
    final_dict["attitude_speakerB"] = data_details[key_B]["attitude_harceleur"]


    #cut into segment of length "segment_length" with overlap, and create the array of features
    t1, t2 = 0-shift_value, segment_length-shift_value
    
    while t2 <= end_time_result and t2 <= end_time_result_speakerB:
        print("Times:", t1, "|", t2)
        
        if(t1 >= 0):
            final_dict["hubert_array"].append(create_hubert_embedding(final_dict["wav_path"]+final_dict["key"]+".wav", t1, t2, processor, audio_encoder, hidden_layer))
            cut = df_result[(df_result["timestamp"] < t2) & (df_result["timestamp"] >= t1)]

            final_dict["hubert_array_speakerB"].append(create_hubert_embedding(final_dict["wav_path"]+final_dict["key_speakerB"]+".wav", t1, t2, processor, audio_encoder, hidden_layer))
            cut_speakerB = df_result_speakerB[(df_result_speakerB["timestamp"] < t2) & (df_result_speakerB["timestamp"] >= t1)]
        
        else: # Handle the case where t1 is less than 0
            first_part = silence_hubert_array
            first_part_speakerB = silence_hubert_array
            second_part = create_hubert_embedding(final_dict["wav_path"]+final_dict["key"]+".wav", 0, t2, processor, audio_encoder, hidden_layer)
            second_part_speakerB = create_hubert_embedding(final_dict["wav_path"]+final_dict["key_speakerB"]+".wav", 0, t2, processor, audio_encoder, hidden_layer)
            
            # Concatenate the silence embedding with the actual audio embedding
            hubert_array = []
            hubert_array_speakerB = []
            for i, hubert_array_layer in enumerate(first_part):
                if hubert_array_layer is not None:
                    hubert_array.append(torch.cat((hubert_array_layer, second_part[i]), dim=1))
                    hubert_array_speakerB.append(torch.cat((first_part_speakerB[i], second_part_speakerB[i]), dim=1))
                else:
                    hubert_array.append(None)
                    hubert_array_speakerB.append(None)

            final_dict["hubert_array"].append(hubert_array)
            final_dict["hubert_array_speakerB"].append(hubert_array_speakerB)


            # Create the cut for the visual features
            second_cut = df_result[(df_result["timestamp"] < t2) & (df_result["timestamp"] >= 0)]
            second_cut_speakerB = df_result_speakerB[(df_result_speakerB["timestamp"] < t2) & (df_result_speakerB["timestamp"] >= 0)]
            
            # Create a DataFrame of zeros with the same columns as second_cut
            first_cut = pd.DataFrame(0, index=range(len(second_cut)), columns=second_cut.columns)
            
            # Concatenate the zero DataFrame with the actual data
            cut = pd.concat([first_cut, second_cut]).reset_index(drop=True)
            cut_speakerB = pd.concat([first_cut, second_cut_speakerB]).reset_index(drop=True)

        ## behaviour
        final_dict["behaviour"].append(cut[visual_features].values)
        final_dict["behaviour_speakerB"].append(cut_speakerB[visual_features].values)

        #time
        final_dict["time_array"].append([t1,t2])
        final_dict["details_time"].append(cut["timestamp"].values)


        # List of label keys for both speakers 
        for label in ["dialog_act", "valence", "dominance", "certainty", "arousal"]:
            final_dict[label].append(cut[label].values)
            final_dict[label+"_speakerB"].append(cut_speakerB[label].values)

        # Speak or not
        final_dict["speak_or_not"].append(cut["bool_speak"].values)
        final_dict["speak_or_not_speakerB"].append(cut_speakerB["bool_speak"].values)

        t1, t2 = round(t1+shift_value,2), round(t2+shift_value,2)

    with open(final_path, 'wb') as f:
        pickle.dump(final_dict, f)
        print("")
    del final_dict

# avant de lancer ce fichier il faut avoir un fichier audio (enregistrement d'une voix ou TTS (avec google par exemple) et extraire les IPU avec sppas)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', help='file name', default='')
    parser.add_argument('-segment', type=int, default=2)
    parser.add_argument('-overlap', type=float, default=0)

    parser.add_argument('-gender', default="F")
    parser.add_argument('-dialogAct', default="Declaration/Explanation")
    parser.add_argument('-valence', default="Neutral")
    parser.add_argument('-arousal', default="Neutral")
    parser.add_argument('-certainty', default="Neutral")
    parser.add_argument('-dominance', default="Neutral")
    args = parser.parse_args()

    #reminder of the different categories
    # categories_certainty = ["silence", "Certain", "Neutral", "Uncertain"]
    # categories_valence = ["silence", "Positive", "Negative", "Neutral"]
    # categories_arousal = ["silence", "Active", "Passive", "Neutral"]
    # categories_dominance = ["silence", "Strong", "Weak", "Neutral"]
    # categories_dialogAct = ["silence", "Declaration/Explanation", "Backchannel", "Agree/Accept" , "Disagree/Disaccept", "Question", "Directive", "Opening/Closing", "Apology", "Thanking"]

    file_name = args.file
    segment_length = args.segment #secondes
    timestep = 0.04
    overlap = args.overlap
    
    path, wav_dir, processed_dir, anno_dir, complete_anno_dir, set_dir = getPath()

    wav_file = join(path, wav_dir, file_name+".wav")
    wav_processed_file = join(path, processed_dir, file_name+".csv")
    wav_origin_processed_file = join(path, processed_dir, "origin", file_name+".csv")
    anno_file = join(path, anno_dir, file_name+".csv")
    value_dialogACt = args.dialogAct.replace("/", "").replace("\\", "")
    #param_file = "_"+value_dialogACt+"_"+args.valence+"_"+args.arousal+"_"+args.certainty+"_"+args.dominance
    param_file=""
    set_file = join(path, set_dir, str(segment_length), file_name+param_file+".p")
    print(set_file)

    createCsvFile(wav_file, wav_origin_processed_file)
    removeOverlapAndAlignStep(wav_origin_processed_file, wav_processed_file, timestep)
    df_anno = create_complete_ipu_file(anno_file, args.dialogAct, args.valence, args.arousal, args.certainty, args.dominance)
    print(df_anno)
    create_set(file_name, set_file, wav_file, wav_processed_file, df_anno, args.gender, segment_length, overlap, timestep)
