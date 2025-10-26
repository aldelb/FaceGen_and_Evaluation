import pandas as pd
import torch
import numpy as np

class NoiseGenerator:

    def __init__(self):
        super().__init__()

    def getNoise(self, data, std=0.1, interval=[0,1]): 
        #return noise of size (batch_size, nb_features (config_file), len (as the audio embedding))
        # we maintain temporal consistency in the generated noise 
        begin_noise = torch.randn_like(data[:,:,0].unsqueeze(2)) * std
        end_noise = torch.randn_like(data[:,:,-1].unsqueeze(2)) * std
        final_noise = torch.clone(begin_noise)
        current = torch.clone(begin_noise)
        step = (end_noise - begin_noise)/(data.shape[2]-1)
        for _ in range(data.shape[2]-2):
            current = current + step
            final_noise = torch.cat((final_noise, current), dim=2)
        final_noise = torch.cat((final_noise, end_noise), dim=2)

        noisy_data = data + final_noise
        #noisy_data = torch.clamp(noisy_data, interval[0], interval[1])
        # print("data", data.shape, data.max().item(), data.min().item(), data.mean().item(), data.std().item())
        # print("noise", final_noise.max().item(), final_noise.min().item(), final_noise.mean().item(), final_noise.std().item())
        # print("Noisy", noisy_data.max().item(), noisy_data.min().item(), noisy_data.mean().item(), noisy_data.std().item())
        return noisy_data.to(data)


def get_noise(data, noise_type):
    noise_types = {
        "mismatched": lambda x: x, #melange les segments dans le code de evaluate.py

        #retourner forme de x qu'avec des 0
        "static": lambda x: np.zeros_like(x), #static agent

        "gaussian": lambda x: x + np.random.normal(0.0, 0.2, size=x.shape), #Ajout d’un bruit gaussien standard

        "partial_gaussian": lambda x, zeta=100: (x + 0.2 * np.random.randn(*x.shape) * ((np.abs(np.arange(x.shape[0])[:, None] - np.random.randint(zeta, max(zeta + 1, x.shape[0] - zeta)))) <= zeta).astype(float)), #Ajout d’un bruit gaussien standard sur quelques frames seulement

        "salt_pepper_zero": lambda x: x * (np.random.rand(*x.shape) > 0.1), #Mise à zéro aléatoire de certains points

        "salt_pepper": lambda x, zeta=0.1: x + np.where((u := np.random.rand(*x.shape)) < zeta, np.where(u < zeta / 2, 0.2, -0.2), 0.0), #modifie aléatoirement une fraction zeta des éléments : la moitié avec +0.2, l'autre moitié avec -0.2, les autres restent inchangés.

        "scaling_noise": lambda x: x * (1.0 + np.random.normal(0.0, 0.2, size=(x.shape[0], 1))), #Applique un bruit multiplicatif local

        "shuffle_time": lambda x: x[np.random.permutation(x.shape[0]), :], #Mélange aléatoire le long du temps (désordre)

        "shuffle_feature": lambda x: x[:, np.random.permutation(x.shape[1])], #Mélange aléatoire le long des caractéristiques (désordre)

        "reverse_time": lambda x: np.flip(x, axis=0), #Inverse la séquence (perte de causalité)

        "temporal_shift": lambda x: np.vstack([np.zeros((10, x.shape[1])), x[:-10, :]]) if x.shape[0] > 10 else x, #Décalage temporel de 10 frames = 0.4 second (perte de causalité)

        "spatial_translation": lambda x: x + 0.5,  #Décalage spatial constant, perturbe légèrement la position (ex: révèle faiblesse du Dice)

        "fake_rhythm": lambda x: x + 0.1 * np.tile(np.sin(np.linspace(0, 4 * np.pi, x.shape[0]))[:, None], (1, x.shape[1])), #Ajoute un motif rythmique faux sinusoidal (ex: WPD)

        "low_energy_noise": lambda x: x + (np.abs(x) < 0.1) * np.random.normal(0.0, 0.2, size=x.shape), #Injection de bruit dans les régions silencieuses (ex: Range Validity)

        "temporal_sparse_interp": lambda x: np.interp(
                                                    np.arange(x.shape[0]),
                                                    np.arange(0, x.shape[0], 10),
                                                    x[::10],
                                                    axis=0
                                                ) if x.ndim == 1 else np.stack([
                                                    np.interp(np.arange(x.shape[0]), np.arange(0, x.shape[0], 10), x[::10, i])
                                                    for i in range(x.shape[1])
                                                ], axis=1)  # Interpolation entre frames, conserve 1 ligne sur 10 et reconstruit la séquence originale par interpolation linéaire (simule une perte de résolution temporelle suivie d'une reconstruction)
        }

    if noise_type not in noise_types:
        raise ValueError(f"Bruit inconnu : {noise_type}")
    
    noisy_data = noise_types[noise_type](data)
    return noisy_data


def get_noise_df(df, noise_type):
    df = df.copy()
    noisy_array = get_noise(df.values, noise_type)
    return pd.DataFrame(noisy_array, columns=df.columns)


def get_noise_torch(data, noise_type):
    noise_types = {
        "mismatched": lambda x: x, #melange les segments dans le code de evaluate.py

        "static": lambda x: torch.zeros_like(x), #static agent

        "gaussian": lambda x: x + 0.2 * torch.randn_like(x), #Ajout d’un bruit gaussien standard

        "partial_gaussian": lambda x, zeta=100: x + 0.2 * torch.randn_like(x) * ((torch.arange(x.shape[0])[:, None] - torch.randint(zeta, x.shape[0]-zeta, (1,))).abs() <= zeta).float(), #Ajout d’un bruit gaussien standard sur quelques frames seulement

        "salt_pepper_zero": lambda x: x * (torch.rand_like(x) > 0.1).float(), #Mise à zéro aléatoire de certains points

        "salt_pepper": lambda x: x + ((torch.rand_like(x) < 0.1).float() * (torch.rand_like(x) < 0.5).float() * 0.2 - (torch.rand_like(x) < 0.1).float() * (torch.rand_like(x) >= 0.5).float() * 0.2), #modifie aléatoirement une fraction zeta des éléments : la moitié avec +0.2, l'autre moitié avec -0.2, les autres restent inchangés.

        "scaling_noise": lambda x: x * (1.0 + 0.2 * torch.randn(x.size(0), 1, x.size(2)).to(x)), #	Applique un bruit multiplicatif local

        "shuffle_time": lambda x: x[:, torch.randperm(x.size(1)), :], #Mélange aléatoire le long du temps (désordre)

        "shuffle_feature": lambda x: x[:, :, torch.randperm(x.size(2))], #Mélange aléatoire le long des caractéristiques (désordre)

        "reverse_time": lambda x: torch.flip(x, dims=[1]), #	Inverse la séquence (perte de causalité)

        "temporal_shift": lambda x: torch.cat([torch.zeros_like(x[:, :10, :]), x[:, :-10, :]], dim=1), #Décalage temporel (perte de causalité)
        
        "spatial_translation": lambda x: x + 0.5,  # Décalage spatial constant (simule un déplacement de tout le corps)

        "fake_rhythm": lambda x: x + 0.1 * torch.sin(torch.linspace(0, 4 * torch.pi, x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)).expand_as(x),  # Ajoute un motif rythmique artificiel (diversité temporelle trompeuse)

        "low_energy_noise": lambda x: x + ((x.abs() < 0.1).float() * 0.2 * torch.randn_like(x)),  # Ajout de bruit dans les régions calmes (perturbe les distributions d’intensité)
        
        "temporal_sparse_interp": lambda x: torch.nn.functional.interpolate(x[:, ::10, :].transpose(1, 2), size=x.size(1), mode="linear", align_corners=True).transpose(1, 2) # Interpolation entre frames, conserve 1 ligne sur 10 et reconstruit la séquence originale par interpolation linéaire (simule une perte de résolution temporelle suivie d'une reconstruction)
    }

    if noise_type not in noise_types:
        raise ValueError(f"Bruit inconnu : {noise_type}")
    
    noisy_data = noise_types[noise_type](data)
    return noisy_data