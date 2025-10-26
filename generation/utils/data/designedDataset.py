from math import floor
import random
import torch
import utils.constants.constants as constants
from torch.utils.data import Dataset

class DesignedDataset1(Dataset):
    def __init__(self, audio, behaviour, speak_or_not, one_hot_labels):
        self.all_audio = audio
        self.all_behaviour = behaviour
        self.all_speak_or_not = speak_or_not
        self.all_one_hot_labels = one_hot_labels
        self.create_index()

    def create_index(self):
        self.listining_index = []
        self.speaking_index = []
        for index, speak_boolean_value in enumerate(self.all_speak_or_not):
            speak_boolean_value = speak_boolean_value.to(dtype=torch.float32)
            if(torch.mean(speak_boolean_value).item() < 0.2): #mostly "listening"
                self.listining_index.append(index) 
            elif(torch.mean(speak_boolean_value) > 0.8): #mostly "speaking"
                self.speaking_index.append(index)
        self.all_index = list(range(len(self.all_audio)))

    def __len__(self):
        return len(self.all_audio)
    

    def extract_criteria(self, chosen_criteria):
        # Initialize feature lists
        r_audio, r_behaviour, f_behaviour  = [], [], []
        r_labels = {current_lab: [] for current_lab in constants.list_of_labels}

        if chosen_criteria == "CLS":# criteria : audio Listening, behav Speaking
            idx_listen_1, idx_speak_1 = random.choice(self.listining_index), random.choice(self.speaking_index)
            r_audio = self.all_audio[idx_listen_1]
            r_behaviour = self.all_behaviour[idx_listen_1]
            f_behaviour = self.all_behaviour[idx_speak_1]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_listen_1]

        elif chosen_criteria == "CSL":# criteria : audio Speaking, behav Listening
            idx_listen_2, idx_speak_2 = random.choice(self.listining_index), random.choice(self.speaking_index)
            r_audio = self.all_audio[idx_speak_2]
            r_behaviour = self.all_behaviour[idx_speak_2]
            f_behaviour = self.all_behaviour[idx_listen_2]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_speak_2]

        elif chosen_criteria == "CMR":# criteria : mix randomly
            idx_mix_audio, idx_mix_behav = random.choice(self.all_index), random.choice(self.all_index)
            r_audio = self.all_audio[idx_mix_audio]
            r_behaviour = self.all_behaviour[idx_mix_audio]
            f_behaviour = self.all_behaviour[idx_mix_behav]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_mix_audio]

        return r_audio, r_behaviour, f_behaviour, r_labels


    def __getitem__(self, i):
        list_of_criteria = constants.designed_targets
        #choose a criteria depending on the index, and extract the corresponding features
        designed_examples = []
        for i_critic in range(constants.n_critics):
            chosen_criteria = list_of_criteria[(i+i_critic) % len(list_of_criteria)]
            designed_examples.append((chosen_criteria, self.extract_criteria(chosen_criteria)))
        return designed_examples
    

class DesignedDataset2(Dataset):
    def __init__(self, prev_audio, audio, behaviour, prev_behaviour, speak_or_not, audio_speakerB, behaviour_speakerB, one_hot_labels):
        self.all_prev_audio = prev_audio
        self.all_audio = audio
        self.all_behaviour = behaviour
        self.all_prev_behaviour = prev_behaviour
        self.all_speak_or_not = speak_or_not
        self.all_audio_speakerB = audio_speakerB
        self.all_behaviour_speakerB = behaviour_speakerB
        self.all_one_hot_labels = one_hot_labels
        self.create_index()

    def create_index(self):
        self.listining_index = []
        self.speaking_index = []
        for index, speak_boolean_value in enumerate(self.all_speak_or_not):
            speak_boolean_value = speak_boolean_value.to(dtype=torch.float32)
            if(torch.mean(speak_boolean_value).item() < 0.2): #mostly "listening"
                self.listining_index.append(index) 
            elif(torch.mean(speak_boolean_value) > 0.8): #mostly "speaking"
                self.speaking_index.append(index)
        self.all_index = list(range(len(self.all_audio)))

    def __len__(self):
        return len(self.all_audio)
    

    def extract_criteria(self, chosen_criteria):
        # Initialize feature lists
        r_audio, r_behaviour, r_prev_behaviour, f_behaviour, r_audio_speakerB, r_behav_speakerB = [], [], [], [], [], []
        r_labels = {current_lab: [] for current_lab in constants.list_of_labels}

        if chosen_criteria == "CLS":# criteria : audio Listening, behav Speaking
            idx_listen_1, idx_speak_1 = random.choice(self.listining_index), random.choice(self.speaking_index)
            r_prev_audio = self.all_prev_audio[idx_listen_1]
            r_audio = self.all_audio[idx_listen_1]
            r_behaviour = self.all_behaviour[idx_listen_1]
            r_prev_behaviour = self.all_prev_behaviour[idx_listen_1]
            f_behaviour = self.all_behaviour[idx_speak_1]
            r_audio_speakerB = self.all_audio_speakerB[idx_listen_1]
            r_behav_speakerB = self.all_behaviour_speakerB[idx_listen_1]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_listen_1]

        elif chosen_criteria == "CSL":# criteria : audio Speaking, behav Listening
            idx_listen_2, idx_speak_2 = random.choice(self.listining_index), random.choice(self.speaking_index)
            r_prev_audio = self.all_prev_audio[idx_speak_2]
            r_audio = self.all_audio[idx_speak_2]
            r_behaviour = self.all_behaviour[idx_speak_2]
            r_prev_behaviour = self.all_prev_behaviour[idx_speak_2]
            f_behaviour = self.all_behaviour[idx_listen_2]
            r_audio_speakerB = self.all_audio_speakerB[idx_speak_2]
            r_behav_speakerB = self.all_behaviour_speakerB[idx_speak_2]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_speak_2]

        elif chosen_criteria == "CMR":# criteria : mix randomly
            idx_mix_audio, idx_mix_behav = random.choice(self.all_index), random.choice(self.all_index)
            r_prev_audio = self.all_prev_audio[idx_mix_audio]
            r_audio = self.all_audio[idx_mix_audio]
            r_behaviour = self.all_behaviour[idx_mix_audio]
            r_prev_behaviour = self.all_prev_behaviour[idx_mix_audio]
            f_behaviour = self.all_behaviour[idx_mix_behav]
            r_audio_speakerB = self.all_audio_speakerB[idx_mix_audio]
            r_behav_speakerB = self.all_behaviour_speakerB[idx_mix_audio]
            for current_lab in constants.list_of_labels:
                r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_mix_audio]
        
        # elif chosen_criteria == "CP":# criteria : mix the prev examples
        #     idx_current, idx_prev = random.choice(self.all_index), random.choice(self.all_index)
        #     r_prev_audio = self.all_prev_audio[idx_prev]
        #     r_audio = self.all_audio[idx_current]
        #     r_behaviour = self.all_behaviour[idx_current]
        #     r_prev_behaviour = self.all_prev_behaviour[idx_prev]
        #     f_behaviour = self.all_behaviour[idx_current]
        #     r_audio_speakerB = self.all_audio_speakerB[idx_current]
        #     r_behav_speakerB = self.all_behaviour_speakerB[idx_current]
        #     for current_lab in constants.list_of_labels:
        #         r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_current]

        # elif chosen_criteria == "CSP":# criteria : mix the speakerB examples
        #     idx_current, idx_b = random.choice(self.all_index), random.choice(self.all_index)
        #     r_prev_audio = self.all_prev_audio[idx_current]
        #     r_audio = self.all_audio[idx_current]
        #     r_behaviour = self.all_behaviour[idx_current]
        #     r_prev_behaviour = self.all_prev_behaviour[idx_current]
        #     f_behaviour = self.all_behaviour[idx_current]
        #     r_audio_speakerB = self.all_audio_speakerB[idx_b]
        #     r_behav_speakerB = self.all_behaviour_speakerB[idx_b]
        #     for current_lab in constants.list_of_labels:
        #         r_labels[current_lab] = self.all_one_hot_labels[current_lab][idx_current]

        return r_prev_audio, r_audio, r_behaviour, r_prev_behaviour, f_behaviour, r_audio_speakerB, r_behav_speakerB, r_labels


    def __getitem__(self, i):
        list_of_criteria = constants.designed_targets
        #choose a criteria depending on the index, and extract the corresponding features
        designed_examples = []
        if(len(list_of_criteria) > 0):
            for i_critic in range(constants.n_critics):
                chosen_criteria = list_of_criteria[(i+i_critic) % len(list_of_criteria)]
                designed_examples.append((chosen_criteria, self.extract_criteria(chosen_criteria)))
        return designed_examples
