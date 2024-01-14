from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations
import csv
import global_

def int_prot(x):
    return int(x, 16) if x.startswith('0x') else int(float(x))

def get_int_time(start_time: str):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    return ts_timestamp

def get_str_time(int_time):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    return datetime.fromtimestamp(int_time).strftime(datetime_format)

def check_label(label_list):
    uni_label = np.unique(label_list)
    
    answer = "BENIGN"
    for label in uni_label:
        label = label.upper()
        if ("BENIGN" not in label):
            return label
    return answer 

def check_flow_label(label):
    label = label.upper()
    if ("BENIGN" not in label):
        return label
    return 'BENIGN'

def make_quantization_dict(train_data, train_key):
    train_label = {}

    train_single_dict = {}
    train_multi_dict = {}

    attack_quantization_single_set = set()
    attack_quantization_multi_set = set()

    train_multi_dict = {}
    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"

        if len(tmp_key.split('_')) == 3:
            if target_ip not in train_multi_dict:
                train_multi_dict[target_ip] = set()
            train_multi_dict[target_ip].add(train_data[idx])
            if label.upper() != 'BENIGN':
                attack_quantization_multi_set.add(train_data[idx])

        if target_ip not in train_label:
            train_label[target_ip] = set()
            
        train_label[target_ip].add(label)

    return train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set

def jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    containment = intersection / union
    return containment

def jaccard_contain(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    containment = intersection / len(set1)
    return containment

def build_inverted_index(pattern_dict):
    inverted_index = defaultdict(list)
    
    for idx, key in enumerate(pattern_dict):
        for word in pattern_dict[key]:
            inverted_index[word].append(key)
    return inverted_index

def evaluate(train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set,\
             test_multi_dict, test_single_dict, test_label, save_file):

    multi_inverted_index = build_inverted_index(train_multi_dict)

    def check_train_label(i):
        for label in train_label[i]:
            if label.upper() != 'BENIGN':
                return label
        
        return 'BENIGN'
    
    def check_test_label(i):
        for label in test_label[i]:
            if label.upper() != 'BENIGN':
                return label
        
        return 'BENIGN'
    
    with open(f"{save_file}", "w", newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["Test IP", "Test IP Label", "Max Sim", "Max IP", "Max IP Label", 'Single Sim', 'Multi Sim'])

        test_key_list = test_multi_dict.keys()

        for ip in tqdm(test_key_list):
            max_sim = (0, 0, 0)
            max_ip = 0
            relevant_indices = set()

            if global_.test_method:
                test_single_filtered_dict = dict()
                test_multi_filtered_dict = dict()

                if ip in test_single_dict:
                    test_single_filtered_dict[ip] = test_single_dict[ip].intersection(attack_quantization_single_set)
                else:
                    test_single_filtered_dict[ip] = set()
                test_multi_filtered_dict[ip] = test_multi_dict[ip].intersection(attack_quantization_multi_set)
                if ip in test_single_dict:
                    for element in test_single_filtered_dict[ip]:
                        relevant_indices.update(single_inverted_index[element])
                if ip in test_multi_dict:
                    for element in test_multi_filtered_dict[ip]:
                        relevant_indices.update(multi_inverted_index[element])
                        
                for train_ip in relevant_indices:
                    if check_train_label(train_ip) == 'BENIGN':
                        continue

                    if train_ip in train_single_dict and ip in test_single_dict:
                        single_sim = jaccard(train_single_dict[train_ip], test_single_dict[ip])
                    else:
                        single_sim = 0
                    if train_ip in train_multi_dict and ip in test_multi_dict:
                        multi_sim = jaccard(train_multi_dict[train_ip], test_multi_dict[ip])
                    else:
                        multi_sim = 0
                    tmp_sim = single_sim * 0.5 + multi_sim * 0.5
                    if tmp_sim > max_sim[0]:
                        max_sim = (tmp_sim, single_sim, multi_sim)
                        max_ip = train_ip
                if max_ip == 0:
                    wr.writerow([ip, check_test_label(ip), '-', '-' , '-', '-', '-'])
                else:
                    wr.writerow([ip, check_test_label(ip), max_sim[0], max_ip, check_train_label(max_ip), max_sim[1], max_sim[2]])

            else:
                if ip in test_single_dict:
                    for element in test_single_dict[ip]:
                        relevant_indices.update(single_inverted_index[element])
                if ip in test_multi_dict:
                    for element in test_multi_dict[ip]:
                        relevant_indices.update(multi_inverted_index[element])
                        
                for train_ip in relevant_indices:               
                    if train_ip in train_single_dict and ip in test_single_dict:
                        single_sim = jaccard(train_single_dict[train_ip], test_single_dict[ip])
                    else:
                        single_sim = 0
                    if train_ip in train_multi_dict and ip in test_multi_dict:
                        multi_sim = jaccard(train_multi_dict[train_ip], test_multi_dict[ip])
                    else:
                        multi_sim = 0
                    tmp_sim = single_sim * 0.5 + multi_sim * 0.5
                    if tmp_sim > max_sim[0]:
                        max_sim = (tmp_sim, single_sim, multi_sim)
                        max_ip = train_ip
                if max_ip == 0:
                    wr.writerow([ip, check_test_label(ip), '-', '-' , '-', '-', '-'])
                else:
                    wr.writerow([ip, check_test_label(ip), max_sim[0], max_ip, check_train_label(max_ip), max_sim[1], max_sim[2]])
