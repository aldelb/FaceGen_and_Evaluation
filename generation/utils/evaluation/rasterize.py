
import utils.constants.features as features
import numpy as np

def classify_au_feature(feature):
    # transforme une valeur continue en une valeur discrete selon 3 catégories
    # 0 : low, 1 : medium, 2 : high
    if feature < 1.5:
        return 0 #low
    elif feature >= 1.5 and feature < 3.5:
        return 1 #medium
    else:
        return 2 #high
    
def calculate_index_activated(list_of_features, l, method):
    # Normalise les features entre 0 et l-1 selon la méthode choisie
    # 1. au_3_categories : 3 catégories (low, medium, high)
    # 2. au_x_categories : l catégories
    # 3. x_categories : normalisation entre 0 et l-1 (valeur entre -1 et 1)

    if(method == "au_3_categories"):
        # 3 classes définies via des seuils fixes
        return [classify_au_feature(feature) for feature in list_of_features]
    elif(method == "au_x_categories"):
        # Discrétisation uniforme entre 0 et 5
        return [min(round(feature*l/5),l-1) for feature in list_of_features]
    elif(method == "x_categories"):
        # mappe des valeurs dans l’intervalle [-1, 1] vers des indices entiers dans [0, l-1]
        return [max(0,min(round((feature+1)*l/2),l-1)) for feature in list_of_features]

def create_one_grid(size):
    # Crée une grille de zeros de taille size[0] x size[1]
    grid = np.zeros((size[0], size[1]), dtype=int)
    return grid

def add_to_grid(grid, x, y, weigth=1):
    # Active  une case de la grille à la position (x,y)
    grid[x][y] = weigth
    return grid

# Rasterise une séquence (dataframe) en une liste de grilles par frame
def create_grid_for_each_frame(df, feature_columns, number_of_feature_categories, method):
    l = number_of_feature_categories
    L = len(feature_columns) 
    grids = []
    for frame in range(len(df)):
        frame_grid = create_one_grid((l,L))
        normalise_feature_frame = calculate_index_activated(df.iloc[frame][feature_columns].values, l, method)
        for index_features, index_intensity in enumerate(normalise_feature_frame):
            frame_grid = add_to_grid(frame_grid, index_intensity, index_features)
        if(frame_grid.sum()!=L):
            print("error in the frame ", frame)
            break
        grids.append(frame_grid)
    return grids

def segment_data(data, segments=3):
    size = len(data)
    return [data[i*size//segments:(i+1)*size//segments] for i in range(segments)]


def activate_symmetric_index(lst):
    #Active l'indice symétrique dans une liste existante.
    length = len(lst)
    for i, value in enumerate(lst):
        if value == 1:
            symmetric_index = length - 1 - i
            lst[symmetric_index] = 1 
            break
    return lst
    
def calculate_total_gac(grids):
    # Aire totale couverte, combien de cases sont activées dans au moins une frame
    union = np.zeros_like(grids[0])
    for grid in grids:
        union = np.logical_or(union, grid)
    return np.sum(union)

def calculate_gac_metrics(ref_grids, test_grids):
    #Calcule le GAC Total, RC et Dice Score entre une séquence test et une séquence référence.

    # Calcul du GAC Total
    total_gac_test = calculate_total_gac(test_grids)
    total_gac_ref = calculate_total_gac(ref_grids)
    
    # Union des grilles pour chaque séquence
    union_test = np.zeros_like(test_grids[0])
    union_ref = np.zeros_like(ref_grids[0])
    
    for grid in test_grids:
        union_test = np.logical_or(union_test, grid)
    for grid in ref_grids:
        union_ref = np.logical_or(union_ref, grid)
    
    # Intersection et calcul des ensembles
    true_positive = np.sum(np.logical_and(union_test, union_ref)) #Aire couverte à la fois par la séquence de référence et la séquence test
    false_positive = np.sum(np.logical_and(union_test, np.logical_not(union_ref))) #Aire couverte par la séquence test mais absente dans la référence 
    false_negative = np.sum(np.logical_and(np.logical_not(union_test), union_ref)) #Aire présente dans la référence mais non couverte par la séquence test
    
    # Calcul des métriques
    rc = (false_positive - false_negative)/total_gac_ref
    dice = (2 * true_positive) / (total_gac_test + total_gac_ref) if (total_gac_test + total_gac_ref) != 0 else 0
    
    if len(ref_grids[0].shape) == 1: #[0] because one grid for each frame
        size_0 = ref_grids[0].shape[0]
        size_1 = 1
    else:
        size_0 = ref_grids[0].shape[0]
        size_1 = ref_grids[0].shape[1]

    return {
        'Total cases': size_0 * size_1,
        'Total GAC Ref': total_gac_ref,
        'Total GAC Test': total_gac_test,
        'True Positive': true_positive,
        'False Positive': false_positive,
        'False Negative': false_negative,
        'Relative Coverage (RC)': rc,
        'Dice Score': dice
    }

# ---------------------------------------------------
# Grids empty creation
# ---------------------------------------------------
def grid_dictionnary(features, columns, type_categorie, number_of_categories):
        sub_grid = {
        "all" : None,
        "first_part": None,
        "second_part": None,
        "third_part": None,
        }

        grids = {"real": sub_grid.copy(), "generated": sub_grid.copy()}
        grids["type"] = features
        grids["columns"] = columns
        grids["type_categories"] = type_categorie
        grids["number_of_categories"] = number_of_categories

        return grids


# ---------------------------------------------------
# Grids creation and rasterization
# ---------------------------------------------------
def create_all_grid(dict_real, dict_generated, type_cat_au, type_cat_head, nb_cat_au, nb_cat_head):
    #create grid for each type of output, the grid will be size (nb_cat, nb_features)
    #one grid per frame
    au_grids = grid_dictionnary("au", features.AU_COLUMNS, type_cat_au, nb_cat_au)
    head_grids = grid_dictionnary("head", features.HEAD_COLUMNS, type_cat_head, nb_cat_head)
    eye_grids = grid_dictionnary("eye", features.EYE_COLUMNS, type_cat_head, nb_cat_head)

    # Rasterize for each file
    for key in dict_real.keys():
        print(key, "length real: ", len(dict_real[key]), "length generated: ", len(dict_generated[key]))
        for i in range(len(dict_real[key])):
            real_frames = dict_real[key][i]
            generated_frames = dict_generated[key][i]
            # On ne garde que le nombre de frames minimum entre les deux séquences
            min_frames = min(len(real_frames), len(generated_frames))
            real_frames = real_frames[:min_frames]
            generated_frames = generated_frames[:min_frames]

            # Pour chaque type d'output
            for features_grid in [au_grids, head_grids, eye_grids]:
                real_grids = create_grid_for_each_frame(real_frames, features_grid["columns"], features_grid["number_of_categories"], features_grid["type_categories"])
                generated_grids = create_grid_for_each_frame(generated_frames, features_grid["columns"], features_grid["number_of_categories"], features_grid["type_categories"])

                if(features_grid["real"]["all"] is None):
                    features_grid["real"]["all"] = np.array(real_grids)
                    features_grid["generated"]["all"] = np.array(generated_grids)
                    features_grid["real"]["first_part"], features_grid["real"]["second_part"], features_grid["real"]["third_part"] = segment_data(np.array(real_grids))
                    features_grid["generated"]["first_part"], features_grid["generated"]["second_part"], features_grid["generated"]["third_part"] = segment_data(np.array(generated_grids))
                else:
                    features_grid["real"]["all"] = np.concatenate((features_grid["real"]["all"], real_grids), axis=0)
                    features_grid["generated"]["all"] = np.concatenate((features_grid["generated"]["all"], generated_grids), axis=0)
                    first_real, second_real, third_real = segment_data(real_grids)
                    first_generated, second_generated, third_generated = segment_data(generated_grids)
                    features_grid["real"]["first_part"] = np.concatenate((features_grid["real"]["first_part"], first_real), axis=0)
                    features_grid["real"]["second_part"] = np.concatenate((features_grid["real"]["second_part"], second_real), axis=0)
                    features_grid["real"]["third_part"] = np.concatenate((features_grid["real"]["third_part"], third_real), axis=0)
                    features_grid["generated"]["first_part"] = np.concatenate((features_grid["generated"]["first_part"], first_generated), axis=0)
                    features_grid["generated"]["second_part"] = np.concatenate((features_grid["generated"]["second_part"], second_generated), axis=0)
                    features_grid["generated"]["third_part"] = np.concatenate((features_grid["generated"]["third_part"], third_generated), axis=0)
    
    return au_grids, head_grids, eye_grids