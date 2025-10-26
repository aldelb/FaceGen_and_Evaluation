import os
from os.path import isfile, join
import time
import opensmile
import numpy as np
import pandas as pd
import pickle
import sys
import librosa
from transformers import Wav2Vec2Processor
import torch
from transformers import HubertModel
import argparse
import parselmouth

from create_set import CLUSTER, OPENSMILE_FEATURES, change_timestep, create_wav2vec_embedding, OPENFACE_FEATURES, putListeningToZero

def getPath(dataset_name, segment_length):
    dic_path = {}
    if(CLUSTER=="jean-zay"):
        general_path = "/lustre/fswork/projects/rech/urk/uln35en/"

        dic_path["silence_path"] = general_path + "/Projets/non-verbal-behaviours-generation/pre_processing/silence/"
        dic_path["silence_wav_path"] = dic_path["silence_path"] + "/silence.wav"
        dic_path["silence_opensmile"] = dic_path["silence_path"] + "/silence.csv"
        dic_path["modelname"] = general_path + "model/hubert-large-ls960-ft/"
        # french_modelname = general_path + "model/exp_w2v2t_fr_hubert_s767/"

        dataset_path = general_path + "raw_data/"+dataset_name+"/"
        init_output_path = dataset_path + "/final_data/"
        output_path = join(init_output_path, str(segment_length))
        os.makedirs(output_path, exist_ok=True)
        
        dic_path["init_output_path"] = init_output_path
        dic_path["output_path"] = output_path
        dic_path["wav_path"] = dataset_path + "audio/full/"
        dic_path["opensmile_path"] = dataset_path + "audio/processed/"
        dic_path["openface_path"] = dataset_path + "video/processed/" 
        dic_path["ipu_path"] = dataset_path + "annotation/processed/ipu/"
        dic_path["ipu_tag_path"] = dataset_path + "annotation/processed/ipu_with_tag/"
        dic_path["details_file"] = dataset_path + "details.xlsx"

    else:
        sys.exit("Error in the cluster name")
    return dic_path

def extract_ipus_and_forced_silences(audio_path, silence_threshold_db=35, min_ipu_duration=0.2):
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    duration = snd.duration

    ipus = []
    is_silent = False
    current_start = None

    for t in range(0, int(duration * 1000)):  # en ms
        time_sec = t / 1000
        try:
            db = intensity.get_value(time_sec)
        except:
            continue

        if db is None or db < silence_threshold_db:
            if not is_silent:
                is_silent = True
                if current_start is not None:
                    current_end = time_sec
                    if current_end - current_start >= min_ipu_duration:
                        ipus.append((current_start, current_end))
                    current_start = None
        else:
            if is_silent or current_start is None:
                current_start = time_sec
                is_silent = False

    # Dernier segment
    if current_start is not None and duration - current_start >= min_ipu_duration:
        ipus.append((current_start, duration))

    # Générer séquence complète avec forçage de # entre chaque IPU
    full_sequence = []
    for i, (start, end) in enumerate(ipus):
        # Ajouter l'IPU
        full_sequence.append(["IPUs", round(start, 3), round(end, 3), f"ipu_{i+1}"])
        # Ajouter le silence entre celui-ci et le suivant
        if i < len(ipus) - 1:
            next_start = ipus[i + 1][0]
            full_sequence.append(["IPUs", round(end, 3), round(next_start, 3), "#"])

    return pd.DataFrame(full_sequence, columns=["ipu", "begin", "end", "speak"])

def create_complete_ipu_file(path_dict, key, label_dict):
    if(not isfile(path_dict["ipu_path"]+key+".csv") and not isfile(path_dict["ipu_tag_path"]+key+".xlsx")):
        wav_file = path_dict["wav_path"] + key + ".wav"
        df_ipu = extract_ipus_and_forced_silences(wav_file)
        df_ipu["bool_speak"] = df_ipu["speak"].transform(lambda x : 0 if x == "#" else 1)
        df_ipu.to_csv(path_dict["ipu_path"]+key+".csv", index=False, header=True)
    else:
        df_ipu = pd.read_csv(path_dict["ipu_path"]+key+".csv")

    if(not isfile(path_dict["ipu_tag_path"]+key+".xlsx")):
        df_ipu["dialog_act"] = label_dict["dialog_act"]
        df_ipu["valence"] = label_dict["valence"]
        df_ipu["arousal"] = label_dict["arousal"]
        df_ipu["certainty"] = label_dict["certainty"]
        df_ipu["dominance"] = label_dict["dominance"]
        df_ipu = df_ipu.drop(columns=['ipu'])
        df_ipu.to_excel(path_dict["ipu_tag_path"]+key+".xlsx", index=False)
    else:
        df_ipu = pd.read_excel(path_dict["ipu_tag_path"]+key+".xlsx")

    return df_ipu


def create_result_df(key, path_dict, timestep, label_dict, interlocutor=False):
    df_ipu = create_complete_ipu_file(path_dict, key, label_dict)

    # Check if alignment file exists or if regeneration is needed
    df_ipu_align_path = join(path_dict["ipu_tag_path"], "align", key+".csv")
    if(not isfile(df_ipu_align_path)):
        # Align the IPU dataframe with the given timestep and save it as a CSV
        df_ipu_align = change_timestep(df_ipu, timestep)
        df_ipu_align.to_csv(df_ipu_align_path, index=False)
    else:
        df_ipu_align = pd.read_csv(df_ipu_align_path)
    # Ensure that the 'timestamp' column is of type float
    df_ipu_align.timestamp = df_ipu_align.timestamp.astype(float)

    df_audio = pd.read_csv(path_dict["opensmile_path"]+key+".csv")[["timestamp"] + OPENSMILE_FEATURES]
    if(interlocutor):
        df_video = pd.read_csv(path_dict["openface_path"]+key+".csv")[["timestamp"] + OPENFACE_FEATURES]

    # Define the path to save or load the merged result
    df_result_path = join(path_dict["init_output_path"], key+".csv")
    # Check if the result file exists or needs regeneration
        # Merge video data with IPU alignment on 'timestamp', drop unnecessary columns
    if(interlocutor):
        df_result = df_video.merge(df_ipu_align, on='timestamp', how='inner')
        df_result = df_result.merge(df_audio, on='timestamp', how='inner')
        print(df_result.columns)
    else:
        df_result = df_audio.merge(df_ipu_align, on='timestamp', how='inner')
    df_result.drop(["begin", "end"], axis=1, inplace=True)
    df_result.to_csv(df_result_path, index=False)

    # Get the last timestamp from the merged result
    end_time_result = df_result["timestamp"].iloc[-1]

    if(interlocutor):
        df_result = putListeningToZero(df_result, ["AU25_r", "AU26_r"]) 

    return df_result, end_time_result



def create_set_one_file(dataset_name, segment_length, overlap, timestep, label_dict, file_name=None, interlocutor=True, attitude_interlocutor="zero"):

    dic_paths = getPath(dataset_name, segment_length)
    
    print("**length", segment_length, "overlap:", overlap, "nombre de frames:", int(segment_length/timestep))

    processor = Wav2Vec2Processor.from_pretrained(dic_paths["modelname"])
    # audio_encoder = HubertModel.from_pretrained(dic_paths["modelname"])
    # audio_encoder.feature_extractor._freeze_parameters()
    # for _, param in audio_encoder.named_parameters():
    #     param.requires_grad = False

    for file in os.listdir(dic_paths["wav_path"]):
        if(".wav" not in file):
            continue

        if(file_name is not None and file_name+".wav" != file):
            continue

        interlocutor = interlocutor
        key_A = file.split(".")[0]
        if("mic" in key_A and interlocutor):
            if("mic1" in key_A):
                key_B = key_A.replace("mic1", "mic2")
            else:
                key_B = key_A.replace("mic2", "mic1")
        else:
            key_B = "no_interlocutor"
            interlocutor = False

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
            
        df_result, end_time_result = create_result_df(key_A, dic_paths, timestep, label_dict)
        #add the columns OPENFACE_FEATURES with value 0 : all the behaviour will be "static" for prev and speakerB, else generated
        for feature in OPENFACE_FEATURES:
            df_result[feature] = 0

        if(interlocutor):
            print("interlocutor")
            df_result_speakerB, end_time_result_speakerB = create_result_df(key_B, dic_paths, timestep, label_dict, interlocutor=True)
            if(attitude_interlocutor=="mix"):
                print("interlocutor, attitude mixed")
                
                # Shuffle all columns except 'timestamp' for all rows
                columns_to_mix = [col for col in df_result_speakerB.columns if col != "timestamp"]
                shuffled = df_result_speakerB[columns_to_mix].sample(frac=1).values
                df_result_speakerB[columns_to_mix] = shuffled
            elif (attitude_interlocutor=="conciliant"):
                print("interlocutor, conciliant")
                df_result_speakerB["AU12_r"] = 4
                df_result_speakerB["AU06_r"] = 4
                df_result_speakerB["AU04_r"] = 0
                df_result_speakerB["AU15_r"] = 0
                
            elif (attitude_interlocutor=="colere"):
                print("interlocutor, colere")
                df_result_speakerB["AU04_r"] = 4
                df_result_speakerB["AU15_r"] = 4
                df_result_speakerB["AU12_r"] = 0
                df_result_speakerB["AU06_r"] = 0
                
            else:
                print("interlocutor, attitude normale")

        else:
            df_result_speakerB = df_result.copy()
            if(attitude_interlocutor=="AU12"):
                print("no interlocutor, attitude AU12")
                df_result_speakerB["AU12_r"] = 4
            elif(attitude_interlocutor=="AU14"):
                print("no interlocutor, attitude AU14")
                df_result_speakerB["AU14_r"] = 4
            else:
                print("no interlocutor, attitude zero")
                df_result_speakerB = df_result.copy()
            df_result_speakerB["bool_speak"] = 0
            silence_values_opensmile = pd.read_csv(dic_paths["silence_opensmile"])[OPENSMILE_FEATURES]
            #keep only one line of silence values
            silence_values_opensmile = silence_values_opensmile.iloc[0:1]
            #put all the line of the features OPENSMILE_FEATURES to the silence values
            for feature in OPENSMILE_FEATURES:
                df_result_speakerB[feature] = silence_values_opensmile[feature].values[0]
            end_time_result_speakerB = end_time_result

        # Store additional speaker information from data_details
        final_dict["gender"] = label_dict["gender"]
        final_dict["role"] = label_dict["role"]
        final_dict["attitude"] = label_dict["attitude"]

        # Store additional speaker B information from data_details
        final_dict["gender_speakerB"] = label_dict["gender"]
        final_dict["role_speakerB"] = label_dict["role"]
        final_dict["attitude_speakerB"] = label_dict["attitude"]

        #cut into segment of length "segment_length" with overlap, and create the array of features
        t1, t2 = 0, segment_length

        #create the previous for the first segments 
        silence_wav2vec = create_wav2vec_embedding(dic_paths["silence_wav_path"], 0, segment_length, processor)
        with open(join(dic_paths["output_path"],"silence_wav2vec.p"), 'wb') as f:
            pickle.dump(silence_wav2vec, f)
        zero_df = pd.DataFrame(0, index=range(int(segment_length/timestep)), columns=df_result.columns)

        #create static behaviour


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
            if(interlocutor):
                final_dict["wav2vec_speakerB"].append(create_wav2vec_embedding(dic_paths["wav_path"]+final_dict["key_speakerB"]+".wav", t1, t2, processor))
            else:
                final_dict["wav2vec_speakerB"].append(silence_wav2vec)

            ## behaviour : openface features
            previous_behaviour = cut[OPENFACE_FEATURES].values
            final_dict["behaviour"].append(previous_behaviour)
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

# avant de lancer ce fichier il faut avoir un fichier audio (enregistrement d'une voix ou TTS (avec google par exemple) et extraire les IPU avec sppas)
def main(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-segment', type=int, default=4)
    parser.add_argument('-overlap', type=int, default=0.4)
    parser.add_argument('-timestep', type=float, default=0.04)
    parser.add_argument('-file_name', type=str, default=None)
    parser.add_argument('-gender', type=str, default="F")
    parser.add_argument("-no_interlocutor", action="store_true", help="Disable interlocutor behaviors")
    parser.add_argument('-attitude_interlocutor', type=str, default="zero")
    args = parser.parse_args()

    dialog_act = "Declaration/Explanation"
    valence = "Neutral"
    arousal = "Neutral"
    certainty = "Neutral"
    dominance = "Neutral"
    gender = "F"
    role = "Temoin"
    attitude = "Neutre"
    label_dict = {"dialog_act": dialog_act, "valence": valence, "arousal": arousal, "certainty": certainty, "dominance": dominance, "gender": gender, "role": role, "attitude": attitude}
    file_name = args.file_name
    segment_length = args.segment #secondes
    timestep = args.timestep #secondes
    overlap_value = args.overlap
    interlocutor = not args.no_interlocutor
    attitude_interlocutor = args.attitude_interlocutor

    begin_time = time.time()
    print("interlocutor1:", interlocutor)
    create_set_one_file("audio_tts", segment_length, overlap_value, timestep, label_dict, file_name, interlocutor, attitude_interlocutor)
    end_time = time.time()
    print(f"Total time taken: {end_time - begin_time} seconds")

if __name__ == "__main__":
    main()