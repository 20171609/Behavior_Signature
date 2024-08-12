from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations
import csv
import global_
from collections import Counter
from collections import deque
def int_prot(x):
    return int(x, 16) if x.startswith('0x') else int(float(x))

def get_int_time(start_time: str):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    return ts_timestamp

def get_str_time(int_time):
    datetime_format = '%Y-%m-%d %H:%M:%S'
    return datetime.fromtimestamp(int_time).strftime(datetime_format)

# 가정 : 한 IP가 두개의 공격을 하지 않음
def check_label(label_list):
    uni_label = np.unique(label_list)

    if len(uni_label) == 1:
        return uni_label[0].upper()

    else:
        for label in uni_label:
            label = label.upper()
            if ("BENIGN" not in label) and ("BACKGROUND" not in label):
                return label
        
        return "BENIGN-MIX"

def check_flow_label(label):
    label = label.upper()
    if ("BENIGN" not in label):
        return label
    return 'BENIGN'

def make_quantization_dict_live_test(train_data, train_key):
    train_label = defaultdict(set)
    train_multi_dict = defaultdict(dict)

    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"
        if ('BENIGN' not in label) and ("BACKGROUND" not in label):
            if train_data[idx] not in train_multi_dict[target_ip]:
                train_multi_dict[target_ip][train_data[idx]] = 0
            train_multi_dict[target_ip][train_data[idx]] += 1
                    
        train_label[target_ip].add(label)

    return train_multi_dict, train_label

def make_quantization_debug_dict(test_data, test_key):
    test_multi_dict = {}
    
    for idx, key in enumerate(tqdm(test_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"

        if target_ip not in test_multi_dict:
            test_multi_dict[target_ip] = set()
        test_multi_dict[target_ip].add(test_data[idx])

    return test_multi_dict