import os
import configparser
import shutil
from datetime import date
from os.path import join
import importlib

import utils.constants.constants as constants


config = configparser.RawConfigParser()

def model_template(task, dataModule, model, trainModel, generateModel):
    """
    Initializes the model based on the task.
    """
    constants.customDataModule = dataModule
    constants.model = model
    if(task == "train"): 
        train = trainModel()
        constants.train_model = train.train_model
    elif(task == "generate" or task=="evaluate" or task=="generate_file"):
        generator = generateModel()
        constants.generate_motion = generator.generate_motion
        constants.generate_latent = generator.generate_latent
    # elif(task == "visualize_data"):
    #     visualize = visualize()
    #     constants.visualize = visualize.visualize_generated_sequences_data
    

def set_model_function(model_name, task):
    """
    Import modules and set the model function based on the model name and task.
    """
    print("model_name", model_name, "task", task)
    
    data_module = getattr(importlib.import_module(f"models.AED_{model_name}.customDataset"), "CustomDataModule")
    train_module = getattr(importlib.import_module(f"models.AED_{model_name}.training"), "TrainModel")
    generate_module = getattr(importlib.import_module(f"models.AED_{model_name}.generating"), "GenerateModel")
    model_module = getattr(importlib.import_module(f"models.AED_{model_name}.model"), "GAN")
    # visualize_data_module = getattr(importlib.import_module(f"models.AED_{model_name}.visualize_data_space"), "VisualizeData")

    # Retrieve the components based on the model name
    model_template(task, data_module, model_module, train_module, generate_module)

    

def read_params(file, task, id=None):
    """
    Reads parameters from the configuration file and sets constants.
    """
    config.read(file)

    # --- Model type params
    constants.model_name =  config.get('MODEL_TYPE','model')
    try:
        constants.french_hubert = config.getboolean('MODEL_TYPE','french_hubert')
    except:
        constants.french_hubert = False
    constants.hidden_state_index = config.getint('MODEL_TYPE','hidden_state_index')
    constants.do_resume = config.getboolean('MODEL_TYPE','resume')
    constants.kernel_size = config.getint('MODEL_TYPE','kernel_size') 
    constants.first_kernel_size = config.getint('MODEL_TYPE','first_kernel_size') 
    constants.dropout =  config.getfloat('MODEL_TYPE','dropout') 
    try: 
        constants.number_of_step = config.getint('MODEL_TYPE','number_of_step')
    except:
        constants.number_of_step = 100

    x1_audio, x2_audio, x3_audio = 64,128,256
    x1_behaviour, x2_behaviour, x3_behaviour = 32,64,128
    x_discr = 64
    constants.generator_weights = {"x1":0, "x2":0, "x3":0}
    constants.discriminator_weights = x_discr #for behav

    constants.x_small_prev_behav = 0 
    try:
        constants.use_small_prev_behav = config.getboolean('MODEL_TYPE','use_small_prev_behav')
        if constants.use_small_prev_behav:
            constants.x_small_prev_behav += 8 #size of the embedding of small previous behaviour    
    except:
        constants.use_small_prev_behav = False

    try:
        constants.use_prev_audio = config.getboolean('MODEL_TYPE','use_prev_audio')
    except:
        constants.use_prev_audio = True
    if(constants.use_prev_audio):
        constants.generator_weights["x1"] += x1_audio
        constants.generator_weights["x2"] += x2_audio
        constants.generator_weights["x3"] += x3_audio
        constants.discriminator_weights += x_discr

    try:
        constants.use_prev_behav = config.getboolean('MODEL_TYPE','use_prev_behav')
    except:
        constants.use_prev_behav = True
    if(constants.use_prev_behav):
        constants.generator_weights["x1"] += x1_behaviour
        constants.generator_weights["x2"] += x2_behaviour
        constants.generator_weights["x3"] += x3_behaviour
        constants.discriminator_weights += x_discr

    try:
        constants.use_audio_speakerB = config.getboolean('MODEL_TYPE','use_audio_speakerB')
    except:
        constants.use_audio_speakerB = True
    if(constants.use_audio_speakerB):
        constants.generator_weights["x1"] += x1_audio
        constants.generator_weights["x2"] += x2_audio
        constants.generator_weights["x3"] += x3_audio
        constants.discriminator_weights += x_discr

    try:
        constants.use_behav_speakerB = config.getboolean('MODEL_TYPE','use_behav_speakerB')
    except:
        constants.use_behav_speakerB = True
    if(constants.use_behav_speakerB):
        constants.generator_weights["x1"] += x1_behaviour
        constants.generator_weights["x2"] += x2_behaviour
        constants.generator_weights["x3"] += x3_behaviour
        constants.discriminator_weights += x_discr

    try:
        constants.use_audio = config.getboolean('MODEL_TYPE','use_audio')
    except:
        constants.use_audio = True
    if(constants.use_audio):
        constants.generator_weights["x1"] += x1_audio
        constants.generator_weights["x2"] += x2_audio
        constants.generator_weights["x3"] += x3_audio
        constants.discriminator_weights += x_discr


    # --- Path params
    datasets = config.get('PATH','datasets')
    constants.datasets = datasets.split(",")
    constants.datasets_properties = config.get("PATH", 'datasets_properties')
    constants.dir_path = config.get('PATH','dir_path')
    constants.data_path = config.get('PATH','data_path')
    constants.saved_path = config.get('PATH','saved_path')
    constants.output_path = config.get('PATH','output_path')
    constants.evaluation_path = config.get('PATH','evaluation_path')
    constants.model_path = config.get('PATH','model_path')
    try :
        constants.finetune =  config.getboolean('PATH','finetune')
        constants.init_model = config.get('PATH','init_model')
    except:
        constants.finetune = False
        constants.init_model = None
    constants.prev_audio_path = join(constants.data_path, constants.datasets[0], "final_data/", constants.datasets_properties, "silence_wav2vec.p")


    # --- Training params
    constants.n_epochs =  config.getint('TRAIN','n_epochs')
    constants.batch_size = config.getint('TRAIN','batch_size')
    constants.d_lr =  config.getfloat('TRAIN','d_lr')
    constants.g_lr =  config.getfloat('TRAIN','g_lr')
    try:
        constants.b1_d = config.getfloat('TRAIN','b1_d')
    except:
        constants.b1_d = 0
    try:
        constants.b1_g = config.getfloat('TRAIN','b1_g')
    except:
        constants.b1_g = 0
    constants.log_interval =  config.getint('TRAIN','log_interval')
    constants.adversarial_coeff = config.getfloat('TRAIN','adversarial_coeff')
    try: 
        constants.adversarial_coeff_att = config.getfloat('TRAIN','adversarial_coeff_att')
    except:
        constants.adversarial_coeff_att = 0.01
    constants.au_coeff = config.getfloat('TRAIN','au_coeff')
    constants.eye_coeff = config.getfloat('TRAIN','eye_coeff')
    constants.pose_coeff = config.getfloat('TRAIN','pose_coeff')
    try :
        constants.gender_coeff = config.getfloat('TRAIN','gender_coeff')
    except : 
        constants.gender_coeff = 0.1
    try:
        constants.reversal_coeff = config.getfloat('TRAIN','reversal_coeff')
    except:
        constants.reversal_coeff = 1

    try:
        constants.evolution_pourcent_generated = config.getboolean('TRAIN', 'evolution_pourcent_generated')
    except: 
        constants.evolution_pourcent_generated = False
        
    constants.pourcent_generated = config.getfloat('TRAIN','pourcent_generated')
    constants.initial_pourcent_generated = constants.pourcent_generated
    constants.step_pourcent = 0.1
    designed_targets = config.get('TRAIN','list_of_targets')
    constants.designed_targets = designed_targets.split(",")
    if(constants.designed_targets == ['']):
        constants.designed_targets = []
    
    try : 
        constants.n_critics = config.getint('TRAIN','n_critic')
    except:
        constants.n_critics = 5

    try : 
        constants.temporal_smoothness = config.getfloat('TRAIN','temporal_smoothness')
    except:
        constants.temporal_smoothness = 0

    try:
        constants.transition_smoothness = config.getfloat('TRAIN','transition_smoothness')
    except:
        constants.transition_smoothness = 0

    try: 
        constants.transition_len = config.getint('TRAIN','transition_len')
    except:
        constants.transition_len = 3
    
    try : 
        constants.std_noise = config.getfloat('TRAIN','noise')
    except:
        constants.std_noise = 0.1

    try : 
        constants.gp_weight = config.getfloat('TRAIN','gp_weight')
    except:
        constants.gp_weight = 50

    try : 
        constants.zero_audio = config.getboolean('TRAIN','zero_audio')
    except:
        constants.zero_audio = False


    # --- Data params
    constants.pose_size = config.getint('DATA','pose_size') 
    constants.eye_size = config.getint('DATA','eye_size')
    constants.pose_t_size = config.getint('DATA','pose_t_size')
    constants.pose_r_size = config.getint('DATA','pose_r_size')
    constants.au_size = config.getint('DATA','au_size') 
    constants.behaviour_size = constants.eye_size + constants.pose_r_size + constants.au_size

    # --- Labels params
    set_labels(task, constants.model_name, config)
    print("list of labels:", constants.list_of_labels, "number of dim labels:", constants.number_of_dim_labels)

    # --- Saved path depending on the task
    if(task == "train" and not constants.do_resume):
        constants.saved_path = create_saved_path(file, id)
        shutil.copy(file, constants.saved_path)
    else:
        constants.saved_path = join(constants.saved_path, constants.model_path)
    
    # --- Function depending on the model and the task
    set_model_function(constants.model_name, task)


############################################################################################################################

def create_saved_path(config_file, id):
    """
    Creates a path for saving files based on the current date and dataset.
    """
    today = date.today().strftime("%d-%m-%Y")
    saved_path = constants.saved_path 
    str_dataset = ""
    for dataset in constants.datasets:
        str_dataset += dataset + "_"

    dir_path = f"{today}_{str_dataset}"
    if(id == "0"):
        i = 1
        while(os.path.isdir(saved_path + dir_path + f"{i}")):
            i = i+1
        dir_path += f"{i}/"
    else:
        dir_path += f"{id}/"
    saved_path += dir_path
    os.makedirs(saved_path, exist_ok=True)
    config.set('PATH','model_path', dir_path)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return saved_path

def set_labels(task, model_name, config):
    """
    Sets label parameters based on the model and task.
    """
    dim_of_labels = {"dialog_act": 10, "valence": 4, "arousal": 4, "certainty": 4, "dominance": 4, "attitude": 4, "gender": 3, "role": 3} #TODO: depend of the dataset None ou speakOnly
    constants.list_of_labels = []
    constants.number_of_dim_labels = 0
    constants.with_labels = False
    
    for label in config.options('LABELS'):
        if(config.getboolean('LABELS',label)):
            constants.number_of_dim_labels += dim_of_labels[label]
            constants.list_of_labels.append(label)
            constants.with_labels = True


    # Visualize all labels and create fake examples even if they are not used in the model
    constants.main_list_for_loading = []
    constants.main_number_of_dim_labels = 0
    for key, value in dim_of_labels.items():
        constants.main_list_for_loading.append(key)
        constants.main_number_of_dim_labels += value