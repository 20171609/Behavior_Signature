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
    train_multi_dict = {}
    attack_quantization_multi_set = set()
    
    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"
        if label != 'BENIGN':
            if target_ip not in train_multi_dict:
                train_multi_dict[target_ip] = []
            train_multi_dict[target_ip].append(train_data[idx])
            attack_quantization_multi_set.add(train_data[idx])

        if target_ip not in train_label:
            train_label[target_ip] = set()
                    
        train_label[target_ip].add(label)

    return train_multi_dict, train_label, attack_quantization_multi_set

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

def evaluate(train_multi_dict, train_label, test_data, test_key, save_file):

    def check_train_label(i):
        for label in train_label[i]:
            if label.upper() != 'BENIGN':
                return label
        
        return 'BENIGN'
    
    train_counter = dict()
    for train_ip in train_multi_dict.keys():
        train_counter[train_ip] = Counter(train_multi_dict[train_ip])
    
    test_multi_dict = dict()
    test_label_dict = dict()
    test_max_sum_dict = dict()
    test_sum_dict = dict()
    test_count_dict = dict()
    test_max_dict = dict()
    denominator = global_.window
    
    for idx,key in tqdm(enumerate(test_key)):
        
        label,ip,file = key.split("+")
        
        key_ = f'{ip}_{file}'
        sig = test_data[idx]
        
        if key_ not in test_multi_dict: # test window list 생성 
            test_multi_dict[key_] = deque([])
            test_count_dict[key_] = dict()
            test_max_sum_dict[key_] = 0
            test_sum_dict[key_] = dict()
            test_max_dict[key_] = 0
            
        test_multi_dict[key_].append(sig)
        
        if sig not in test_count_dict[key_]: #test count dict 생성 
            test_count_dict[key_][sig]=0
        test_count_dict[key_][sig] += 1
        
        if key_ not in test_label_dict: #test label dict 생성 
            test_label_dict[key_] = label
        elif label.upper()!='BENIGN':
            test_label_dict[key_]=label
        
        if len(test_multi_dict[key_])==global_.window: # case1. 처음 10개 유사도 계산 
            sig_list = test_multi_dict[key_]
            
            for train_ip in train_counter.keys():    
                sum = 0
                if train_ip not in test_sum_dict[key_]:
                    test_sum_dict[key_][train_ip]=0
                
                for s in set(sig_list):
                    if s in train_counter[train_ip]:
                        sum += min(sig_list.count(s),train_counter[train_ip][s])
                test_sum_dict[key_][train_ip] = sum
                
                if  sum > test_max_sum_dict[key_]:
                    test_max_dict[key_] = train_ip
                    test_max_sum_dict[key_]= sum
                    
        elif len(test_multi_dict[key_])>global_.window:
            
            out_ = test_multi_dict[key_].popleft()
            test_count_dict[key_][out_] -= 1
            
            for train_ip in train_counter.keys():
                sum = test_sum_dict[key_][train_ip]
                
                if (out_ in train_counter[train_ip]) and (test_count_dict[key_][out_]< train_counter[train_ip][out_]):
                    sum -= 1
                if (sig in train_counter[train_ip]) and (test_count_dict[key_][sig] <= train_counter[train_ip][sig]):
                    sum += 1
                    
                test_sum_dict[key_][train_ip] = sum
                    
                if  sum > test_max_sum_dict[key_]:
                    test_max_dict[key_] = train_ip
                    test_max_sum_dict[key_]= sum
                    
            if test_count_dict[key_][out_] == 0:
                del test_count_dict[key_][out_]   
        # if count < 50  :
        #     print(test_multi_dict[key_])
        #     print(test_max_dict[key_])
        #     print(test_sim_dict[key_])
        #     print(test_count_dict[key_])
        #     print()
        #     print()
        
    with open(f"{save_file}", "w", newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["Test IP", "Test IP Label", "Max IP", "Max IP Label", "Max Sim"])

        for ip in test_multi_dict.keys():
            max_ip = test_max_dict[ip]
            if max_ip == 0:
                wr.writerow([ip, test_label_dict[ip], '-', '-' , '-'])
            else:
                wr.writerow([ip, test_label_dict[ip], max_ip, check_train_label(max_ip), test_max_sum_dict[ip]/denominator])
