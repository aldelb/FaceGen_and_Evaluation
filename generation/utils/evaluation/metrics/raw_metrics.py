import math
from math import ceil
import os
from os.path import isdir, isfile, join
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dtaidistance import dtw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import tensor
from torchmetrics.classification import Dice
from aeon.distances import dtw_alignment_path, dtw_distance
import torch

import utils.constants.features as features
import utils.constants.constants as constants
import utils.evaluation.rasterize as rasterize


def cut_min_frames(real_frames, generated_frames):
    min_frames = min(len(real_frames), len(generated_frames))
    real_frames = real_frames[:min_frames]
    generated_frames = generated_frames[:min_frames]
    return real_frames, generated_frames

#---------------------------------
# MOTION MEASURES
#---------------------------------
def init_motion_measures(dict_data):
    dt = features.DT # 25 fps
    df_acceleration = pd.DataFrame(0.0, index=dict_data.keys(), columns=features.ALL_FEATURES)
    df_jerk = pd.DataFrame(0.0, index=dict_data.keys(), columns=features.ALL_FEATURES)

    for key in dict_data.keys(): #for each video
        print(key, "number of segments : ", len(dict_data[key]))
        for segment in dict_data[key]:
            data = segment
            result_acceleration_file = []
            result_jerk_file = []

            for feature in features.ALL_FEATURES: #for each feature
                # First derivative : speed-velocity
                velocity = data[feature].diff() / dt
                # Second derivative : acceleration
                acceleration = velocity.diff() / dt
                acceleration_final = np.mean(acceleration[2:].abs())
                # Third derivative : jerk
                jerk = acceleration.diff() / dt
                jerk_final = np.mean(jerk[3:].abs())

                result_acceleration_file.append(acceleration_final)
                result_jerk_file.append(jerk_final)

            # Add the results to the dataframe for each segment
            df_acceleration.loc[key] += result_acceleration_file
            df_jerk.loc[key] += result_jerk_file

        # Average the results for each video on the number of segments
        df_acceleration.loc[key] /= len(dict_data[key])
        df_jerk.loc[key] /= len(dict_data[key])
                
    # Add the mean of each feature and each video to the dataframe
    df_acceleration.loc['mean'] = df_acceleration.mean()
    df_acceleration["mean"] = df_acceleration.mean(axis=1)
    df_jerk.loc['mean'] = df_jerk.mean()
    df_jerk["mean"] = df_jerk.mean(axis=1)

    return df_acceleration, df_jerk

def compute_motion_measures(dict_real, dict_generated, path_evaluation):
        print("*"*10, "MOTION DATA", "*"*10)
        real_acceleration, real_jerk = init_motion_measures(dict_real)
        real_acceleration.to_excel(join(path_evaluation, "real_acceleration.xlsx"))
        real_jerk.to_excel(join(path_evaluation, "real_jerk.xlsx"))


        generated_acceleration, generated_jerk = init_motion_measures(dict_generated)
        generated_acceleration.to_excel(join(path_evaluation, "generated_acceleration.xlsx"))
        generated_jerk.to_excel(join(path_evaluation, "generated_jerk.xlsx"))

        diff_acceleration = abs(generated_acceleration - real_acceleration)
        diff_jerk = abs(generated_jerk - real_jerk)
        # diff_acceleration.to_excel(self.path_evaluation+"diff_acceleration.xlsx")
        # diff_jerk.to_excel(self.path_evaluation+"diff_jerk.xlsx")

        diff_motion_summary = pd.DataFrame(columns=features.ALL_FEATURES)
        mean_diff_acceleration = diff_acceleration.loc['mean']
        mean_diff_jerk = diff_jerk.loc['mean']

        diff_motion_summary.loc['mean_acceleration'] = mean_diff_acceleration
        diff_motion_summary.loc['mean_jeark'] = mean_diff_jerk

        diff_motion_summary.to_excel(join(path_evaluation, "diff_motion_summary.xlsx"))

#---------------------------------
# VELOCITY HISTOGRAMS AND HELLIGER DISTANCE
#---------------------------------

def compute_hellinger_distance(dict_real, dict_generated, path_evaluation):
    print("*"*10, "VELOCITY HISTOGRAM", "*"*10)
    dt = features.DT # 25 fps
    df_velocity_histo = pd.DataFrame(0.0, index=dict_real.keys(), columns=features.ALL_FEATURES)
    
    for key in dict_real.keys(): #for each video
        print(key, "number of segments : ", len(dict_real[key]))
        for i, segment in enumerate(dict_real[key]): #for each segment
            real_frames = dict_real[key][i]
            generated_frames = dict_generated[key][i]
            real_frames, generated_frames = cut_min_frames(real_frames, generated_frames)
            for feature in features.ALL_FEATURES: #for each feature
                
                real_velocity = real_frames[feature].diff() / dt
                generated_velocity = generated_frames[feature].diff() / dt
                
                if np.array_equal(real_velocity[1:], generated_velocity[1:]):
                    distance = 0.0
                else:
                    # Calcul des histogrammes normalisés
                    velocity_histo = velocity_histogram(real_velocity[1:], generated_velocity[1:])
                    # Calcul de la distance de Hellinger
                    distance = hellinger_distance(velocity_histo[0], velocity_histo[1])
                if(math.isnan(distance)):
                    print(real_velocity[1:].sum(), generated_velocity[1:].sum())
                    print("distance : ", distance, "feature : ", feature, "video : ", key, "segment : ", i)

                df_velocity_histo.loc[key, feature] += distance
        
        df_velocity_histo.loc[key] /= len(dict_real[key])

    df_velocity_histo.loc['mean'] = df_velocity_histo.mean()
    df_velocity_histo["mean"] = df_velocity_histo.mean(axis=1)
    df_velocity_histo.to_excel(join(path_evaluation, "hellinger_distance.xlsx"))

# Fonction de distance de Hellinger
def hellinger_distance(hist1, hist2):
    return np.sqrt(1.0 - np.sum(np.sqrt(hist1 * hist2)))

def velocity_histogram(real_velocity, generated_velocity):
    # Calcul des histogrammes normalisés
    bins = np.linspace(0, 3, 100)  # Intervalle [0, 3] avec 30 bins
    hist1, _ = np.histogram(real_velocity, bins=bins, density=True)
    hist2, _ = np.histogram(generated_velocity, bins=bins, density=True)

    epsilon = 1e-10  # Petite valeur pour éviter les zéros
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon

    # Renormaliser après l'ajout de epsilon
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # # Visualisation des histogrammes
    # plt.figure(figsize=(3, 3), dpi=100)
    # plt.hist(real_velocity, bins=bins, alpha=0.5, label="Real", density=True)
    # plt.hist(generated_velocity, bins=bins, alpha=0.5, label="Generated", density=True)
    # plt.legend()
    # plt.title(features_name)
    # plt.xlabel("Vitesse")
    # plt.ylabel("Densité")
    # pdf.savefig()
    # plt.close()

    return hist1, hist2

#---------------------------------
# DTW
#---------------------------------

def create_dtw(dict_real, dict_generated, path_evaluation):
    print("*"*10,"DTW", "*"*10)
    dist_df = pd.DataFrame(0.0, index=dict_real.keys(), columns=features.ALL_FEATURES)
    for key in dict_real.keys(): #for each video
        print(key, "number of segments : ", len(dict_real[key]))
        for i, segment in enumerate(dict_real[key]): #for each (1min) segment
            dist_file = []
            real_frames_video = dict_real[key][i]
            generated_frames_video = dict_generated[key][i]
            real_frames_video, generated_frames_video = cut_min_frames(real_frames_video, generated_frames_video)

            for feature in features.ALL_FEATURES: #for each feature
                #print("*"*2,feature, "*"*2)
                real_frames = real_frames_video[feature].to_numpy()
                generated_frames = generated_frames_video[feature].to_numpy()
                distance = dtw_distance(real_frames, generated_frames)
                dist_file.append(distance)

            dist_df.loc[key] += dist_file #add all the values for each segment to each others

        dist_df.loc[key] /= len(dict_real[key]) #average the values by the number of segments

    dist_df.loc['mean'] = dist_df.mean()
    dist_df["mean"] = dist_df.mean(axis=1)
    dist_df.to_excel(join(path_evaluation, "dtw.xlsx"))

#---------------------------------
# WPD
#---------------------------------
def create_wpd(real_sequences, generated_sequences, path_evaluation):
    print("*"*10,"WPD", "*"*10)
    # Save the results
    columns = ["WPD"]
    results = pd.DataFrame(columns=columns)
    print("shapes : ", real_sequences.shape, generated_sequences.shape)
    real_wpd = compute_wpd(real_sequences, Swpd=200, runs=5, seed=42)
    generated_wpd = compute_wpd(generated_sequences, Swpd=200, runs=5, seed=42)

    results.loc["real"] = [real_wpd]
    results.loc["generated"] = [generated_wpd]
    results.loc["diff"] = [abs(real_wpd - generated_wpd)]

    # Save the results
    results.to_excel(join(path_evaluation,"wpd.xlsx"))

def compute_wpd(sequences: np.ndarray, Swpd: int = 200, runs: int = 5, seed: int = 42) -> float:
    """
    Compute Warping Path Deviation (WPD).

    Args:
        sequences (np.ndarray): (N, T, D) array of generated sequences
        Swpd (int): number of samples to compare in each run
        runs (int): number of iterations to average over
        seed (int): random seed for reproducibility

    Returns:
        float: average WPD score (↑ = plus de diversité temporelle)
    """
    # convert torch tensor to numpy if needed
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.detach().cpu().numpy()

    np.random.seed(seed)
    N = len(sequences)
    Swpd = min(Swpd, N)

    if sequences.ndim == 4:
        sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], -1)  # (N, T, Dflat)

    wpd_values = []

    for _ in range(runs):
        idx_1 = np.random.choice(N, size=Swpd, replace=False)
        idx_2 = np.random.choice(N, size=Swpd, replace=False)

        S1 = sequences[idx_1]
        S2 = sequences[idx_2]

        for i in range(Swpd):
            path, _ = dtw_alignment_path(S1[i], S2[i])  # path = list of (i, j)
            path = np.asarray(path)

            deviation = (np.sqrt(2) / (2 * len(path))) * np.sum(np.abs(path[:, 0] - path[:, 1]))
            wpd_values.append(deviation)

    return float(np.mean(wpd_values))



#---------------------------------
# DICE
#---------------------------------

def create_dice_score_and_coverage(dict_real, dict_generated, path_evaluation):
    print("*"*10,"DICE SCORE", "*"*10)
    # Create all grids
    au_grids, head_grids, eye_grids = rasterize.create_all_grid(dict_real, dict_generated, "au_x_categories", "x_categories", 25, 25)
    print("Grids created")

    # Save the results
    metrics_columns = ['Total cases', "Total GAC Test", "Total GAC Ref", "True Positive", "False Positive", "False Negative", "Relative Coverage (RC)", "Dice Score"]
    results = pd.DataFrame(columns=metrics_columns)

    # Compute the dice and RC score
    for features_grid in [au_grids, head_grids, eye_grids]: # For each group of features
        metrics_all = rasterize.calculate_gac_metrics(features_grid["real"]["all"], features_grid["generated"]["all"])
        results.loc[features_grid["type"]+"_all"] = [metrics_all[feature] for feature in metrics_columns]
        first_metrics = rasterize.calculate_gac_metrics(features_grid["real"]["first_part"], features_grid["generated"]["first_part"])
        results.loc[features_grid["type"]+"_first_part"] = [first_metrics[feature] for feature in metrics_columns]
        second_metrics = rasterize.calculate_gac_metrics(features_grid["real"]["second_part"], features_grid["generated"]["second_part"])
        results.loc[features_grid["type"]+"_second_part"] = [second_metrics[feature] for feature in metrics_columns]
        third_metrics = rasterize.calculate_gac_metrics(features_grid["real"]["third_part"], features_grid["generated"]["third_part"])
        results.loc[features_grid["type"]+"_third_part"] = [third_metrics[feature] for feature in metrics_columns]

        for i, current_feature in enumerate(features_grid["columns"]): # for each feature
            current_real_grids = [real_grid[:,i] for real_grid in features_grid["real"]["all"]]
            current_generated_grids = [generated_grid[:,i] for generated_grid in features_grid["generated"]["all"]]
            metrics_feature = rasterize.calculate_gac_metrics(current_real_grids, current_generated_grids)
            results.loc[current_feature] = [metrics_feature[feature] for feature in metrics_columns]

    # Save the results
    results.to_excel(join(path_evaluation,"dice_coverage.xlsx"))


#---------------------------------
# RANGE VALIDITY
#---------------------------------


def create_range_validity(dict_real, dict_generated, path_evaluation):
    print("*"*10,"RANGE VALIDITY", "*"*10)

    # Save the results
    columns = [
        "Total",
        "ref_Total_0", "ref_Total_1", "ref_Total_2",
        "test_Total_0", "test_Total_1", "test_Total_2",
        "Same", "Same_0", "Same_1", "Same_2",
        "Same/Total", "accuracy_0", "accuracy_1", "accuracy_2",
    ]
    results = pd.DataFrame(columns=columns)

    # Create all grids
    len_au_grids = 3
    len_other_grids = 7 # Must be odd to permit the symetric treatment
    cat_others = [[2,3,4], [1,5], [0,6]] # High négatif // high positif, take into account the symetric treatment
    
    au_grids, head_grids, eye_grids = rasterize.create_all_grid(dict_real, dict_generated, "au_3_categories", "x_categories", len_au_grids, len_other_grids) #avant abs

    # Compute the range validity per frame
    for features_grid in [au_grids, head_grids, eye_grids]: #pour chaque type de features

        for i, current_feature in enumerate(features_grid["columns"]): #pour chaque colonnes

            current_real_grids = [real_grid[:,i] for real_grid in features_grid["real"]["all"]]
            current_generated_grids = [generated_grid[:,i] for generated_grid in features_grid["generated"]["all"]]

            same, different = 0, 0
            vp = [0,0,0] #vrai positifs # low, medium, high
            fn = [0,0,0] #false négatifs
            fp = [0,0,0] #false positifs
            ref_count = [0,0,0]
            test_count = [0,0,0]

            for frame in range(len(current_real_grids)):
                current_real_frame = current_real_grids[frame]
                current_generated_frame = current_generated_grids[frame]

                #pas besoin non, le fait d'avoir repartie les index en 3 catégorie ca suffit je pense
                #if current_feature in features.SYMETRIC_FEATURES :
                    #current_real_frame = metrics.activate_symmetric_index(current_real_frame)
                    #current_generated_frame = rasterize.activate_symmetric_index(current_generated_frame)

                # Trouver les indices activés
                real_index = np.argmax(current_real_frame)
                test_index = np.argmax(current_generated_frame)

                # Mapping AU ou autres
                if features_grid["type"] == "au":
                    real_class = real_index
                    test_class = test_index
                else:
                    real_class = (
                        0 if real_index in cat_others[0]
                        else 1 if real_index in cat_others[1]
                        else 2
                    )
                    test_class = (
                        0 if test_index in cat_others[0]
                        else 1 if test_index in cat_others[1]
                        else 2
                    )

                # Count the number of elements in each category
                ref_count[real_class] += 1
                test_count[test_class] += 1

                if real_class == test_class:
                    same += 1
                    vp[real_class] += 1
                else:
                    different += 1
                    fn[real_class] += 1 #we create the false_negative (really real_class but predicted test_class)
                    fp[test_class] += 1 #we create the false_positive (really real_class but predicted test_class)
                            

            total = same + different
            total_0 = ref_count[0]
            total_1 = ref_count[1]
            total_2 = ref_count[2]

            #result with f1 score
            result = (same) / (total)
            precision_0 = (vp[0]) / (vp[0] + fp[0]) if (vp[0] + fp[0]) != 0 else 0
            precision_1 = (vp[1]) / (vp[1] + fp[1]) if (vp[1] + fp[1]) != 0 else 0
            precision_2 = (vp[2]) / (vp[2] + fp[2]) if (vp[2] + fp[2]) != 0 else 0

            recall_0 = (vp[0]) / (vp[0] + fn[0]) if (vp[0] + fn[0]) != 0 else 0
            recall_1 = (vp[1]) / (vp[1] + fn[1]) if (vp[1] + fn[1]) != 0 else 0
            recall_2 = (vp[2]) / (vp[2] + fn[2]) if (vp[2] + fn[2]) != 0 else 0
            
            result_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) != 0 else 0
            result_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) != 0 else 0
            result_2 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2) if (precision_2 + recall_2) != 0 else 0

            results.loc[current_feature] = [
                total,
                ref_count[0], ref_count[1], ref_count[2],
                test_count[0], test_count[1], test_count[2],
                same, vp[0], vp[1], vp[2],
                result, result_0, result_1, result_2
            ]

    #add a mean row
    results.loc["mean"] = results.mean(numeric_only=True)

    #save the results
    results.to_excel(join(path_evaluation,"range_validity.xlsx"))

#---------------------------------
# CURVES
#---------------------------------

def plot_figure(real_signal, generated_signal, pdf, features_name):
    x_real = range(len(real_signal))
    x_gen = range(len(generated_signal))
    plt.figure(figsize=(3, 3), dpi=100)
    plt.title(features_name)
    plt.plot(x_gen, generated_signal, label="generated", alpha=0.5, rasterized=True)
    plt.plot(x_real, real_signal, label="real", alpha=0.8, rasterized=True)
    plt.legend()
    pdf.savefig()
    plt.close()

def create_curve(real_frames, generated_frames, path_evaluation):
    print("*"*10, "GENERAL CURVE", "*"*10)
    with PdfPages(join(path_evaluation , "curve.pdf")) as pdf:
        for feature in features.ALL_FEATURES : 
            plot_figure(real_frames[feature], generated_frames[feature], pdf, feature)
    
def create_curves_video(dict_real, dict_generated, path_evaluation):
    print("*"*10,"VIDEO CURVE", "*"*10) 
    for key in dict_real.keys():
        real_frames_video = dict_real[key]
        generated_frames_video = dict_generated[key]
        with PdfPages(join(path_evaluation, key + "_curve.pdf")) as pdf:
            for feature in features.ALL_FEATURES :
                real_frames = real_frames_video[feature]
                gened_frames = generated_frames_video[feature]
                plot_figure(real_frames, gened_frames, pdf, feature)

#---------------------------------
# PCA
#---------------------------------

def create_pca(real_frames, generated_frames, path_evaluation):
    print("*"*10,"PCA", "*"*10)
    with PdfPages(join(path_evaluation, "PCA.pdf")) as pdf:
        for cle, feature in features.TYPES_OUTPUT.items():
            #print("*"*5,cle, "*"*5)
            if(cle != "clignement"):
                compute_pca(real_frames[feature], generated_frames[feature], pdf, cle)


def compute_pca(real_frames, gened_frames, pdf, features_name = ""):
    scaler = StandardScaler()
    scale_real = scaler.fit(real_frames)

    X_real = scale_real.transform(real_frames)
    X_gened = scale_real.transform(gened_frames)
    mypca = PCA(n_components=2, random_state=1) # calculate the three major components
    pca_real = mypca.fit(X_real)

    #pca in graphs
    data_real = pca_real.transform(X_real)
    data_generated = pca_real.transform(X_gened)
    col_list = ['principal component 1', 'principal component 2']
    df_real = pd.DataFrame(data = data_real, columns=col_list)
    df_generated = pd.DataFrame(data = data_generated, columns=col_list)
    indicesToKeep = df_generated.index

    plt.figure(figsize=(3, 3), dpi=100)
    plt.title('pca_'+features_name)
    plt.scatter(df_real.loc[indicesToKeep, 'principal component 1'], df_real.loc[indicesToKeep, 'principal component 2'], label='Real data', rasterized=True)
    plt.scatter(df_generated.loc[indicesToKeep, 'principal component 1'], df_generated.loc[indicesToKeep, 'principal component 2'], label='Generated data', alpha=0.7, rasterized=True)
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.legend()
    pdf.savefig() # saves the current figure into a pdf page
    plt.close()