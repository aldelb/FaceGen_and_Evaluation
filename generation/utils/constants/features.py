#---------------------------------
# CONSTANTS
#---------------------------------

OUTPUT_COLUMNS = ["timestamp", "gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", 
                "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz", 
                "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


ALL_FEATURES = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry", "pose_Rz", 
                "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

TYPES_OUTPUT = {"gaze" : ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"],
                "pose" : ["pose_Rx", "pose_Ry", "pose_Rz"],
                "sourcils" : ["AU01_r", "AU02_r", "AU04_r"],
                "yeux" : ["AU05_r","AU06_r", "AU07_r", "AU45_r"],
                "bouche_sup" : ["AU09_r", "AU10_r", "AU12_r", "AU14_r"],
                "ouv_bouche": ["AU23_r", "AU25_r", "AU26_r"],
                "bouche_inf" : ["AU15_r", "AU17_r", "AU20_r"]
                }

AU_COLUMNS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", 
"AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

IMPORTANT_AUS = ["AU01_r", "AU02_r", "AU04_r", "AU12_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"] #les AUs qui se percoivent le plus sur l'agent virtuel

HEAD_COLUMNS = ["pose_Rx", "pose_Ry", "pose_Rz"]

EYE_COLUMNS = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]

SYMETRIC_FEATURES = ["pose_Ry", "pose_Rz", "gaze_0_x", "gaze_0_z", "gaze_1_x", "gaze_1_z"]

OPENFACE_FEATURES = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

OPENSMILE_FEATURES = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", 
                      "mfcc3_sma3", "mfcc4_sma3", "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", 
                      "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz", "F2bandwidth_sma3nz", 
                      "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz"]

AUS_WT_SPEAK = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU45_r"]

INDEX_AUS_WT_SPEAK = [OPENFACE_FEATURES.index(au) for au in AUS_WT_SPEAK]

SEGMENT_LENTGH = 4 #in seconds

OVERLAP = 0.4 #in seconds

DT = 0.04 #in seconds