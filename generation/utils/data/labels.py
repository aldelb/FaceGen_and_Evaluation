import torch
from collections import Counter
import random


categories = {
    "dialog_act": ["silence", "Declaration/Explanation", "Backchannel", "Agree/Accept", "Disagree/Disaccept", "Question", "Directive", "Opening/Closing", "Apology", "Thanking"],
    "valence": ["silence", "Positive", "Negative", "Neutral"],
    "arousal": ["silence", "Active", "Passive", "Neutral"],
    "dominance": ["silence", "Strong", "Weak", "Neutral"],
    "certainty": ["silence", "Certain", "Neutral", "Uncertain"],
    "gender": ["H", "F", "silence"],
    "small_gender": ["H", "F"],
    "attitude": ["Neutre", "Colere_froide", "Colere_chaude", "Conciliant"],
    "small_attitude": ["None", "Neutral", "Colere_chaude", "Conciliant"],
    "role": ["Contexte", "Temoin", "Harceleur"],
}

label_to_index = {}
index_to_label = {}

for type, labels in categories.items():
    label_to_index[type] = {label: i for i, label in enumerate(sorted(labels))}
    index_to_label[type] = {i: label for label, i in label_to_index[type].items()}


def get_label_to_index(type):
    return label_to_index[type]

def get_index_to_label(type):
    return index_to_label[type]

def label_to_one_hot(current_label, type):
    if(current_label == 0):
        current_label = "silence"
    label_to_index_type = label_to_index[type]
    one_hot = torch.zeros(len(label_to_index_type))
    one_hot[label_to_index_type[current_label]] = 1
    return one_hot

def one_hot_to_label(one_hot, type):
    index_to_label_type = index_to_label[type]
    num_index = torch.argmax(one_hot).item()
    return index_to_label_type[num_index]

def one_hot_to_index(one_hot, type):
    label_to_index_type = label_to_index[type]
    return label_to_index_type[one_hot_to_label(one_hot, type)]

def other_label(categories, current_label_list, type):
    new_list = []
    for one_hot_current_label in current_label_list:
        current_label = one_hot_to_label(one_hot_current_label, type)
        list_wt_label = [l for l in categories if l != current_label]
        new_label = random.choice(list_wt_label)
        new_list.append(label_to_one_hot(new_label, type))
    return torch.stack(new_list)

def get_other_label(label_list, type):
    return other_label(categories[type], label_list, type)



def get_labels(type):
    return categories[type]

def get_color(type):
    if type == "dialog_act":
        return {"silence": "grey", "Declaration": "blue", "Backchannel":"yellow", "Agree/accept":"green" , "Disagree/disaccept":"red", "Question":"purple", "Directive":"black" , "Non-understanding":"orange", "Opening":"pink", "Apology":"brown", "Thanking":"olive"}
    elif type == "valence":
        return {"silence": "grey", "Positive": "green", "Negative":"red", "Neutral":"blue"}
    elif type == "arousal":
        return {"silence": "grey", "Active": "green", "Passive":"red", "Neutral":"blue"}
    elif type == "dominance":
        return {"silence": "grey", "Strong": "green", "Weak":"red", "Neutral":"blue"}
    elif type == "certainty":
        return {"silence": "grey", "Certain": "green", "Neutral":"red", "Uncertainty":"blue"}
    elif type == "gender":
        return {"silence": "grey", "H": "red", "F" : "green"}
    elif type == "small_gender":
        return {"H": "red", "F" : "green"}
    elif type == "attitude":
        return {"Neutre": "grey", "Colere_froide":"yellow", "Colere_chaude":"red", "Conciliant":"blue"}
    elif type == "role":
        return {"Contexte": "grey", "Temoin":"yellow", "Harceleur":"red"}


def get_maj_label(labels):
        # Count the number of occurrences of each label
        label_counts = Counter(labels)
        # Find the majority label
        majority_label = max(label_counts, key=label_counts.get)
        # Calculate the percentage presence of the majority label
        percentage_majority = label_counts[majority_label] / len(labels) * 100
        # If there's something other than silence, we'll take something else.
        if majority_label == "silence" and percentage_majority < 100:
            majority_label = Counter(labels).most_common(2)[1][0]
            #second_percentage_majority = label_counts[second_majority_label] / len(labels) * 100
        return majority_label


def supress_silence_index(data, one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    supress_index = []
    for i in range(len(raw_labels_list)):
        if "silence" in raw_labels_list[i]:
            supress_index.append(i)
    tensor = data.clone()
    masque = torch.ones(data.size(0), dtype=torch.bool).to(tensor)
    masque[supress_index] = False
    tensor_without_silence = torch.index_select(tensor, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    one_hot_labels_without_silence = torch.index_select(one_hot_labels_list, dim=0, index=torch.nonzero(masque).squeeze()).to(tensor)
    labels_without_silence = [raw_labels_list[i] for i in range(len(raw_labels_list)) if i not in supress_index]
    # print("******len after supress silence**********")
    # print(len(tensor_without_silence), len(labels_without_silence))
    return tensor_without_silence, one_hot_labels_without_silence

def get_no_silence_index_from_one_hot(one_hot_labels_list, type):
    raw_labels_list = [one_hot_to_label(label, type) for label in one_hot_labels_list]
    index_no_silence = [index for index, element in enumerate(raw_labels_list) if element != "silence"]
    return index_no_silence
