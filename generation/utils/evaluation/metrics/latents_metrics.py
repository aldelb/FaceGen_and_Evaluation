from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np



#------------------------------
# DIVERSITY METRICS
#------------------------------

def compute_coverage(real_latents: np.ndarray, gen_latents: np.ndarray, k: int = 5) -> float:
    """
    Compute the COVERAGE metric from Naeem et al. (2020)

    Args:
        real_latents (np.ndarray): embeddings des exemples réels (N_real, D)
        gen_latents (np.ndarray): embeddings des exemples générés (N_gen, D)
        k (int): nombre de voisins dans le graphe des vrais exemples

    Returns:
        coverage (float): valeur entre 0 et 1
    """
    print("*"*10, "COVERAGE", "*"*10)
    # Étape 1 : distances vraies ↔ générées
    distance_real_to_gen = pairwise_distances(real_latents, gen_latents)  # (N_real, N_gen)

    # Étape 2 : rayon k-NN dans les exemples réels
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 pour éviter soi-même
    nn.fit(real_latents)
    dist_real_knn, _ = nn.kneighbors(real_latents)
    kth_distances = dist_real_knn[:, -1]  # (N_real,)

    # Étape 3 : distance au plus proche généré
    min_distances = np.min(distance_real_to_gen, axis=1)  # (N_real,)

    # Étape 4 : couverture = combien de vrais sont "couverts" par les générés
    covered = min_distances < kth_distances  # booléen
    coverage = np.mean(covered.astype(np.float32))

    print("Coverage:", coverage)
    return coverage


def compute_mms(real_latents: np.ndarray, gen_latents: np.ndarray) -> float:
    """
    Compute Minimum Matching Similarity (MMS).
    
    Args:
        real_latents (np.ndarray): embeddings des données réelles (N_real, D)
        gen_latents (np.ndarray): embeddings des données générées (N_gen, D)
        
    Returns:
        float: distance moyenne au plus proche réel (plus petit = mieux)
    """
    print("*"*10, "MMS", "*"*10)
    k = 1
    # if np.array_equal(real_latents, gen_latents):
    #     # Cas où on veut MMS interne (ex: MMS(x, x)), on ignore le point lui-même
    #     k = 2

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(real_latents)

    distances, _ = nn.kneighbors(gen_latents, return_distance=True)  # (N_gen, k)

    if distances.shape[1] > 1:
        distances = distances[:, 1]  # Ignorer soi-même

    print("MMS:", np.mean(distances))
    return float(np.mean(distances))


def compute_apd(latents: np.ndarray, Sapd: int = 200, runs: int = 5, seed: int = 42) -> float:
    """
    Compute Average Pairwise Distance (APD) as defined in Naeem et al. (2020).
    
    Args:
        latents (np.ndarray): (N, D) vecteurs latents des échantillons générés
        Sapd (int): nombre d'échantillons à tirer dans chaque set aléatoire
        runs (int): nombre de répétitions pour estimer la diversité
        seed (int): graine pour reproductibilité
    
    Returns:
        float: score APD (plus grand = mieux pour la diversité)
    """
    print("*"*10, "APD", "*"*10)
    np.random.seed(seed)
    N = len(latents)
    Sapd = min(Sapd, N)
    
    scores = []
    for _ in range(runs):
        idx_1 = np.random.choice(N, size=Sapd, replace=False)
        idx_2 = np.random.choice(N, size=Sapd, replace=False)
        V = latents[idx_1]
        V_prime = latents[idx_2]

        # distance euclidienne entre chaque paire V[i] - V'[i]
        dists = np.linalg.norm(V - V_prime, axis=1)
        scores.append(np.mean(dists))
    print("APD:", np.mean(scores))
    return np.mean(scores)


#------------------------------
# ACCURACY METRICS
#------------------------------

def compute_density(real_latents: np.ndarray, gen_latents: np.ndarray, k: int = 5) -> float:
    """
    Compute the DENSITY metric from Naeem et al. (2020)

    Args:
        real_latents (np.ndarray): embeddings des vrais (N_real, D)
        gen_latents (np.ndarray): embeddings des générés (N_gen, D)
        k (int): nombre de voisins pour les vrais

    Returns:
        density (float): score DENSITY
    """
    print("*"*10, "DENSITY", "*"*10)
    # Matrice (N_real, N_gen) : distance de chaque vrai vers chaque généré
    distance_real_to_gen = pairwise_distances(real_latents, gen_latents)  # (N_real, N_gen)

    # k+1-ème plus proche voisin pour chaque vrai (pour avoir un "radius")
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(real_latents)
    real_knn_dists, _ = nn.kneighbors(real_latents)
    kth_distances = real_knn_dists[:, -1]  # (N_real,)

    # Pour chaque généré, combien de vrais sont "suffisamment proches"
    within_radius = (
        distance_real_to_gen < np.expand_dims(kth_distances, axis=1)
    ).sum(axis=0)

    density = (1 / k) * np.mean(within_radius.astype(np.float32))
    print("Density:", density)
    return density


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

def compute_fgd(X_real, X_fake):
    """
    Compute the Frechet distance between two sets of latents.
    """
    print("*"*10, "FGD", "*"*10)
    mu_real = np.mean(X_real, axis=0)
    mu_fake = np.mean(X_fake, axis=0)
    sigma_real = np.cov(X_real, rowvar=False)
    sigma_fake = np.cov(X_fake, rowvar=False)

    # Compute the Frechet distance
    fgd = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    print("FGD:", fgd)
    return fgd