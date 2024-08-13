from collections import Counter, defaultdict
import global_
import csv
from profiling import b_profiling
import glob
from tqdm.auto import tqdm
import pickle
import os

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




def evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file):
    def check_train_label(i):
        label_set = set()
        for ip_ in i:
            label = ip_.split("_")[0]
            label_set.add(label)
        return label_set
    
    def check_test_label(i):
        if len(test_label[i]) == 1:
            return list(test_label[i])[0]

        for label in test_label[i]:
            if ('BENIGN' not in label) and ("BACKGROUND" not in label):
                return label

        return "BENIGN-MIX"

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
            # if relevant_indices:
            #     print(relevant_indices)
            test_sig = set(test_multi_dict[ip]).intersection(attack_quantization_multi_set)
            
            max_sim = 0 
            max_ip = set()
                        
            for test_block in test_list_n:
                if check_sig(test_block,test_sig):
                    for train_ip in relevant_indices:
                        if ip == '_'.join(train_ip.split('_')[1:]):
                            continue
                        tmp = '_'.join(train_ip.split('_')[1:])
                        
                        if ip == tmp:
                            continue
                        similarity = bag_similarity_counter(test_block,train_multi_dict_c[train_ip])
                        
                        if max_sim == similarity:
                            max_ip.add(train_ip)
                        elif max_sim < similarity:
                            max_sim =similarity
                            max_ip=set([train_ip])

            if len(max_ip) == 0:
                wr.writerow([ip, check_test_label(ip), '-', '-' , '-'])
            else:
                wr.writerow([ip, check_test_label(ip), max_ip, check_train_label(max_ip), max_sim])


def test_no_live(save_file, test_path, parameter, min_data, dataset_path, ignore_background, log, add_src, count_prot, attack, train_multi_dict,  train_label, attack_quantization_multi_set):
    print("test profiling 시작")
    b_profiling(test_path, "test", parameter, min_data, dataset_path, ignore_background)
    print("Test 끝")

    test_raw = []
    test_key = []

    folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'

    # 'test_feature'로 시작하는 모든 파일 찾기
    test_ffiles = glob.glob(os.path.join(folder, 'test_feature*'))
    test_ffiles.sort()
    for file in test_ffiles:
        with open(file, 'rb') as f:
            test_raw += pickle.load(f)
    
    # 'test_key'로 시작하는 모든 파일 찾기
    test_kfiles = glob.glob(os.path.join(folder, 'test_key*'))
    test_kfiles.sort()
    for file in test_kfiles:
        with open(file, 'rb') as f:
            test_key += pickle.load(f)

    test_data = log.multi_transform(test_raw)
    test_src = []
    test_prot = []

    if add_src:
        test_ffiles_src = glob.glob(os.path.join(folder, 'test_srcflag*'))
        test_ffiles_src.sort()
        for file in test_ffiles_src:
            with open(file, 'rb') as f:
                test_src += pickle.load(f)

        test_data = [f"{test}{src}" for test, src in zip(test_data, test_src)]

    if count_prot:
        test_ffiles_prt = glob.glob(os.path.join(folder, 'test_protflag_*'))
        test_ffiles_prt.sort()
        for file in test_ffiles_prt:
            with open(file, 'rb') as f:
                test_prot += pickle.load(f)

        test_data = [f"{test}{prt}" for test, prt in zip(test_data, test_prot)]


    #         with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
    #             test_data = pickle.load(f)

    # with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'wb') as f:
    #     pickle.dump(test_data,f)
    
    test_multi_dict, test_label = make_quantization_test_dict(test_data, test_key)
    
    # with open(f"./debug_data/{dataset_path}/{parameter}/test_multi_dict_attack{attack}.pkl", 'wb') as f:
    #     pickle.dump(test_multi_dict,f)

    evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file)