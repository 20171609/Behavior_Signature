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

def make_quantization_dict_window(train_data, train_key, train_window):
    train_label = {}
    train_multi_dict = {}
    train_count_dict = {}
    train_queue_dict = {}

    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"
        target_ip_idx = f'{target_ip}_{idx}'
        data = train_data[idx]
        
        if label != 'BENIGN':
            if target_ip_idx not in train_multi_dict: 
                train_multi_dict[target_ip_idx] = []
            if target_ip not in train_count_dict:
                train_count_dict[target_ip] = dict()
            if target_ip not in train_queue_dict:
                train_queue_dict[target_ip] = deque()

            if data not in train_count_dict[target_ip]:
                train_count_dict[target_ip][data]=0 
            
            train_count_dict[target_ip][data]+=1    
            train_queue_dict[target_ip].append(data)
            
            if len(train_queue_dict[target_ip]) == train_window:
                
                # if train_window <= 10:
                #     if train_count_dict[target_ip] != train_multi_dict[target_ip_idx]:
                #         train_multi_dict[target_ip_idx] = train_count_dict[target_ip]
                # else:
                train_multi_dict[target_ip_idx].append(train_count_dict[target_ip])
                train_count_dict[target_ip] = dict()
                train_queue_dict[target_ip] = deque()
            # elif len(train_queue_dict[target_ip]) > train_window:
            #     out_ = train_queue_dict[target_ip].popleft()
            #     train_count_dict[target_ip][out_]-=1
                
            #     if train_count_dict[target_ip][out_]==0:
            #         del train_count_dict[target_ip][out_]
            #     if data in train_count_dict[target_ip]:
            #         train_count_dict[target_ip][data]+=1
            #     else:
            #         train_count_dict[target_ip][data]=1
            #     train_multi_dict[target_ip_idx].append(train_count_dict[target_ip])
                
            if target_ip not in train_label:
                train_label[target_ip] = set()
            train_label[target_ip].add(label)
    
    return train_multi_dict, train_label
def make_quantization_dict_window_notreal(train_data, train_key, train_window):
    train_label = {}
    train_multi_dict = {}
    train_count_dict = {}
    train_queue_dict = {}
  

    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"
        target_ip_idx = f'{target_ip}_{idx}'
        data = train_data[idx]
        
        if label != 'BENIGN':
            if target_ip_idx not in train_multi_dict: 
                train_multi_dict[target_ip_idx] = []
            if target_ip not in train_count_dict:
                train_count_dict[target_ip] = dict()
            if target_ip not in train_queue_dict:
                train_queue_dict[target_ip] = deque()

            if data not in train_count_dict[target_ip]:
                train_count_dict[target_ip][data]=0 
            
            train_count_dict[target_ip][data]+=1    
            train_queue_dict[target_ip].append(data)
            
            if len(train_queue_dict[target_ip]) == train_window:
                
                # if train_window <= 10:
                #     if train_count_dict[target_ip] != train_multi_dict[target_ip_idx]:
                #         train_multi_dict[target_ip_idx] = train_count_dict[target_ip]
                # else:
                train_multi_dict[target_ip_idx].append(train_count_dict[target_ip])
                train_count_dict[target_ip] = dict()
                train_queue_dict[target_ip] = deque()
            # elif len(train_queue_dict[target_ip]) > train_window:
            #     out_ = train_queue_dict[target_ip].popleft()
            #     train_count_dict[target_ip][out_]-=1
                
            #     if train_count_dict[target_ip][out_]==0:
            #         del train_count_dict[target_ip][out_]
            #     if data in train_count_dict[target_ip]:
            #         train_count_dict[target_ip][data]+=1
            #     else:
            #         train_count_dict[target_ip][data]=1
            #     train_multi_dict[target_ip_idx].append(train_count_dict[target_ip])
                
            if target_ip not in train_label:
                train_label[target_ip] = set()
            train_label[target_ip].add(label)
    
    return train_multi_dict, train_label
def make_quantization_dict(train_data, train_key):
    train_label = {}
    train_multi_dict = {}
    attack_quan_set = set()
    for idx, key in enumerate(tqdm(train_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"
        if label != 'BENIGN':
            if target_ip not in train_multi_dict:
                train_multi_dict[target_ip] = []
            train_multi_dict[target_ip].append(train_data[idx])
            attack_quan_set.add(train_data[idx])
        if target_ip not in train_label:
            train_label[target_ip] = set()
                    
        train_label[target_ip].add(label)

    return train_multi_dict, train_label ,attack_quan_set
def make_quantization_debug_dict(test_data, test_key):
    test_multi_dict = {}
    
    for idx, key in enumerate(tqdm(test_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{label}_{tmp_key.split('_')[0]}_{file}"

        if target_ip not in test_multi_dict:
            test_multi_dict[target_ip] = set()
        test_multi_dict[target_ip].add(test_data[idx])

    return test_multi_dict

def make_quantization_test_dict(test_data, test_key):
    test_label = {}
    test_multi_dict = {}
    
    for idx, key in enumerate(tqdm(test_key)):
        label, tmp_key, file = key.split('+')
        target_ip = f"{tmp_key.split('_')[0]}_{file}"

        if target_ip not in test_multi_dict:
            test_multi_dict[target_ip] = []
        test_multi_dict[target_ip].append(test_data[idx])
        
        if target_ip not in test_label:
            test_label[target_ip] = set()
                    
        test_label[target_ip].add(label)

    return test_multi_dict,test_label

def bag_similarity(list1, list2):

    counter1 = Counter(list1)
    counter2 = Counter(list2)

    intersection = sum((counter1 & counter2).values())
    similarity = intersection / global_.test_window
    
    return similarity


def bag_similarity_counter(list1, counter2):

    counter1 = Counter(list1)

    intersection = sum((counter1 & counter2).values())
    similarity = intersection / global_.test_window
    
    return similarity

def build_inverted_index(pattern_dict):
    inverted_index = defaultdict(set)
    
    for idx, key in enumerate(pattern_dict):
        for word in pattern_dict[key]:
            inverted_index[word].add(key)
    return inverted_index

def evaluate_realtime(train_multi_dict, train_label, test_data, test_key, save_file): # train window 로 train split여부 결정 가능 

    def check_train_label(i):
        for label in train_label[i]:
            if label.upper() != 'BENIGN':
                return label

    train_counter = dict()
    train_window = global_.train_window
    if train_window:
        train_counter = train_multi_dict
    else:
        for train_ip in train_multi_dict.keys():
            train_counter[train_ip] = Counter(train_multi_dict[train_ip])
    
    test_multi_dict = dict()
    test_label_dict = dict()
    test_max_sum_dict = dict()
    test_sum_dict = dict()
    test_count_dict = dict()
    test_max_dict = dict()
    
    for idx,key in tqdm(enumerate(test_key), total=len(test_key)):
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
        
        if len(test_multi_dict[key_])==global_.test_window: # case1. 처음 10개 유사도 계산 
            sig_list = test_multi_dict[key_]

            for train_ip in train_counter.keys():    
                sum_ = 0
                if train_ip not in test_sum_dict[key_]:
                    test_sum_dict[key_][train_ip]=0
                
                for s in set(sig_list):
                    if s in train_counter[train_ip]:
                        sum_ += min(sig_list.count(s),train_counter[train_ip][s])
                test_sum_dict[key_][train_ip] = sum_

                if  sum_ > test_max_sum_dict[key_]:
                    test_max_dict[key_] = train_ip
                    test_max_sum_dict[key_]= sum_

        elif len(test_multi_dict[key_])>global_.test_window:
            
            out_ = test_multi_dict[key_].popleft()
            test_count_dict[key_][out_] -= 1

            for train_ip in train_counter.keys():
                    
                sum_ = test_sum_dict[key_][train_ip]

                if out_ != sig:
                    if (out_ in train_counter[train_ip]) and (test_count_dict[key_][out_] < train_counter[train_ip][out_]):
                        sum_ -= 1
                    if (sig in train_counter[train_ip]) and (test_count_dict[key_][sig] <= train_counter[train_ip][sig]):
                        sum_ += 1

                test_sum_dict[key_][train_ip] = sum_

                if  sum_ > test_max_sum_dict[key_]:
                    test_max_dict[key_] = train_ip
                    test_max_sum_dict[key_]= sum_

            if test_count_dict[key_][out_] == 0:
                del test_count_dict[key_][out_]

    denominator = global_.test_window
    with open(f"{save_file}", "w", newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["Test IP", "Test IP Label", "Max IP", "Max IP Label", "Max Sim"])

        for ip in test_multi_dict.keys():
            max_ip = test_max_dict[ip]
            
            if max_ip == 0:
                wr.writerow([test_label_dict[ip] + '_' + ip, test_label_dict[ip], '-', '-' , '-'])
            else:
                max_ip2 = max_ip
                if len(max_ip.split("_"))==2:
                    max_ip2 = max_ip.split("_")[0]
                wr.writerow([test_label_dict[ip] + '_' + ip, test_label_dict[ip], max_ip2, check_train_label(max_ip), test_max_sum_dict[ip]/denominator])

def evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file):
    
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
    def check_attack_dict(ip):
        relevant_indices = set()
        signature_ = test_multi_dict[ip]
        
        for s in signature_:
            if s in multi_inverted_index:
                relevant_indices.update(multi_inverted_index[s])
        return relevant_indices
    
    def n_gram(input_list, n):
        return [input_list[i:i+n] for i in range(len(input_list) - n + 1)]
    
    def check_sig(sig_set,list_):
        set_ = set(list_)
        for s in set_:
            if s in sig_set:
                return True
        return False
    
    multi_inverted_index = build_inverted_index(train_multi_dict)
    
    if global_.train_window:
        train_multi_dict_n = dict()
        for train_ip in train_multi_dict:
            train_multi_dict_n[train_ip] = n_gram(train_multi_dict[train_ip],global_.train_window)
    else:
        train_multi_dict_c = dict()
        for train_ip in train_multi_dict:
            train_multi_dict_c[train_ip] = Counter(train_multi_dict[train_ip])
    
    with open(f"{save_file}", "w", newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["Test IP", "Test IP Label", "Max IP", "Max IP Label", "Max Sim"])

        for ip in tqdm(test_multi_dict.keys()):
            
            test_list_n = n_gram(test_multi_dict[ip],10)
            unique_list = list(set(tuple(sorted(element)) for element in test_list_n))
            result = [list(t) for t in unique_list]
            test_list_n = result
            
            relevant_indices = check_attack_dict(ip)
            
            test_sig = set(test_multi_dict[ip]).intersection(attack_quantization_multi_set)
            
            max_sim = 0 
            max_ip = 0
            
            if global_.train_window:
                for test_block in (test_list_n):
                    if check_sig(test_block,test_sig):
                        test_counter = Counter(test_block)
                        for train_ip in relevant_indices:
                            for train_block in train_multi_dict_n[train_ip]:
                                if check_sig(train_block,test_sig):
                                    similarity = bag_similarity_counter(train_block, test_counter)
                                    if max_sim < similarity:
                                        max_sim =similarity
                                        max_ip = train_ip
                                    if max_sim == 1:
                                        break
            else:
                for test_block in test_list_n:
                    if check_sig(test_block,test_sig):
                        for train_ip in relevant_indices:             
                            similarity = bag_similarity_counter(test_block,train_multi_dict_c[train_ip])
                            
                            if max_sim < similarity:
                                max_sim =similarity
                                max_ip = train_ip
                                if max_sim == 1:
                                        break
            if max_ip == 0:
                wr.writerow([ip, check_test_label(ip), '-', '-' , '-'])
            else:
                wr.writerow([ip, check_test_label(ip), max_ip, check_train_label(max_ip), max_sim])
