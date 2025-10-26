
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
import sklearn

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('whitegrid')

#disable debbuging API for faster training
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
# enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

seed = 2855
random.seed(seed)
np.random.seed(seed)
sklearn.utils.check_random_state(seed)
torch.manual_seed(seed)
pl.seed_everything(seed)

from utils.constants.constants_utils import read_params
from evaluate import Evaluate
from train import Train
from generate import Generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-id', help='Path to save result and models', default="0")
    parser.add_argument('-task', help='choose a task beetween "train", "generate", "evaluate"', default="train")

    #for generation and evaluation
    parser.add_argument('-file', help='which file we want to generate (one by one)', default=None)
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=1000)

    #for raw evaluation
    parser.add_argument('-raw', action='store_true')

    #for latent evaluation
    parser.add_argument('-latent', action='store_true')

    #for visualization
    parser.add_argument('-visu', action='store_true')

    parser.add_argument('-noise_name', help='choose a type of noise', default="gaussian")
    


    parser.add_argument('-labels', default='', help='label(s) to visualize in the existing one [dialog_act, valence, arousal, certainty, dominance, attitude, gender]')

    args = parser.parse_args()

    task = args.task
    read_params(args.params, task, args.id)

    #-----------------------------------
    # Training of the model
    #-----------------------------------
    if(task == "train"):
        train = Train()

    #-----------------------------------
    # Generation of the dataset
    #-----------------------------------
    elif(task == "generate"):
        generate = Generate(args, "test")
        generate = Generate(args, "val")

    #-----------------------------------
    # Generation of a particular file
    #-----------------------------------
    elif(task == "generate_file"):
        generate = Generate(args, "audio_tts")

    #-----------------------------------
    # Evaluation with ref dataset cut in two parts
    #-----------------------------------
    elif(task == "evaluate_ref"): 
        Evaluate("val", args, ref=True)
        Evaluate("test", args, ref=True)

    #-----------------------------------
    # Evaluation with ref dataset vs ref dataset with different noises
    #-----------------------------------
    elif(task == "evaluate_noise"):
        Evaluate("val", args, noise=True)
        Evaluate("test", args, noise=True)

    #-----------------------------------
    # Evaluation with generated dataset
    #-----------------------------------
    elif(task == "evaluate"):
        Evaluate("val", args)
        #Evaluate("test", args)
        