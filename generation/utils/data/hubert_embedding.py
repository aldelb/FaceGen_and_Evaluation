import torch
from transformers import HubertModel

import utils.constants.constants as constants

class HubertEmbedding:

    def __init__(self):
        super().__init__()
        self.audio_encoder = HubertModel.from_pretrained("/lustre/fswork/projects/rech/urk/uln35en/model/hubert-large-ls960-ft/")
        constants.audio_dim = self.audio_encoder.encoder.config.hidden_size #audio_dim = 1024


    def create_hubert_embedding(self, x):
        # Figer les paramètres du feature extractor
        self.audio_encoder.feature_extractor._freeze_parameters()
        for _, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        embedding = self.audio_encoder(x, output_hidden_states=True)
        x = embedding.hidden_states[constants.hidden_state_index] #24 layers in total if hubert large
        x = x[:, :100*2, :] #keep only the 4 seconds
        return x