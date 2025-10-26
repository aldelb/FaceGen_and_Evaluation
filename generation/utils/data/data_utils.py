import torch
import utils.constants.constants as constants
from sklearn.preprocessing import MinMaxScaler

def get_pose_type_index():
    index_eye = range(constants.eye_size)
    index_pose = range(constants.eye_size, constants.eye_size + constants.pose_r_size)
    index_au = range(constants.eye_size + constants.pose_r_size, constants.eye_size + constants.pose_r_size + constants.au_size)
    return index_eye, index_pose, index_au

def scale_from_scratch(x_array, range="tanh"):
    #creation of the scaler
    if(range=="tanh"):
        minMaxScaler = MinMaxScaler((-1,1))
    elif(range=="sigmoid"):
        minMaxScaler = MinMaxScaler((0,1))
    if(len(x_array.shape) < 3):
        scaler = minMaxScaler.fit(x_array) 
    else:
        scaler = minMaxScaler.fit(x_array.view(-1, x_array.size()[2])) 
    #application of the scaler
    x_scaled = scale(x_array, scaler)
    return x_scaled, scaler

def scale(x, scaler):
    if(len(x.shape) < 3):
        x_scaled = scaler.transform(x)
    else:
        x_scaled = torch.empty(size=(x.size()[0], x.size()[1], x.size()[2]))
        for i in range(x.size()[0]):
            x_scaled[i] = torch.tensor(scaler.transform(x[i]))     
    return x_scaled
   

def rescale(x, scaler):
    if(len(x.shape) < 3):
        x_rescaled = scaler.inverse_transform(x)
    else:
        x_rescaled = torch.empty(size=(x.size()[0], x.size()[1], x.size()[2]))
        for i in range(x.size()[0]):
            x_rescaled[i] = torch.round(torch.tensor(scaler.inverse_transform(x[i].cpu())), decimals=4)
    return x_rescaled

def format_data(inputs_audio, targets):
    target_eye, target_pose_r, target_au = separate_openface_features(targets)
    return inputs_audio.squeeze(1), targets, target_eye, target_pose_r, target_au

def format_targets(targets):
    target_eye, target_pose_r, target_au = separate_openface_features(targets)
    return target_eye, target_pose_r, target_au

def separate_openface_features(targets):
    index_eye = range(0, constants.eye_size)
    target_eye = targets[:,:,index_eye]

    index_target_pose_r = range(constants.eye_size, constants.eye_size + constants.pose_r_size)
    target_pose_r = targets[:,:,index_target_pose_r]

    index_au = range(constants.eye_size + constants.pose_r_size, constants.eye_size + constants.pose_r_size + constants.au_size)
    target_au = targets[:,:,index_au]

    return target_eye, target_pose_r, target_au
    
def reshape_output(target_eye, target_pose_r, target_au, output_scaler):
    outs = torch.cat((target_eye, target_pose_r, target_au), 2)
    outs = rescale(outs, output_scaler)
    return outs

def concat_with_labels(x, len_x, labels, l_dim=1, c_dim=2):
    for label_type in constants.list_of_labels:
        redim_label = torch.repeat_interleave(labels[label_type].unsqueeze(2), len_x, dim=l_dim)
        x = torch.cat([x, redim_label], dim=c_dim)
    return x

def concat_x_labels(x, labels, l_dim=1, c_dim=2):
    len_x = x.size()[l_dim]
    for label_type in constants.list_of_labels:
        redim_label = torch.repeat_interleave(labels[label_type].unsqueeze(2), len_x, dim=l_dim)
        x = torch.cat([x, redim_label], dim=c_dim)
    return x

def putListeningToZero(df, features):
    new_df = df.copy()
    index_listening_behavior = df.where(df['bool_speak'] == 0).dropna().index
    new_df.loc[index_listening_behavior, features] = 0
    return new_df
    
