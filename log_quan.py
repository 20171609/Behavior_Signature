import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp
import global_
from collections import Counter

class log_Pattering:
    def __init__(self, ignore_idx=[0,1,2], n_log_ = 1.2, n_jobs = 1):
        
        self.ignore_idx = ignore_idx
        self.n_log = n_log_
        self.boundary_dict = {}
        
    def find_min_frequency_value(self,lst):
    
        frequencies = Counter(lst)# 리스트의 빈도 계산
        
        min_frequency = min(frequencies.values())# 빈도 중에서 최소값 찾기
        
        min_frequency_elements = [key for key, value in frequencies.items() if value == min_frequency]
        # 최소 빈도값을 가진 원소들 추출
        
        min_value = min(min_frequency_elements) # 최소 빈도값을 가진 원소들 중에서 최솟값 찾기
        return min_value
    
    def make_boundary(self,logN_,feature_list):
        logN = logN_
        
        #min_data = min(feature_list)
        max_data = max(feature_list)
        #make original_list
        original_list = [0]

        tmp=1
        i=1
        while(tmp<=max_data):
            tmp = logN**i
            i+=1
            original_list.append(tmp)
        len_ = len(original_list)
        
        #make real_boundary
        real_boundary = [0]
        for idx,scale in enumerate(original_list):
                
            if idx==(len_-1):
                continue
            left = scale
            right = original_list[idx+1]
        
            filtered_array = feature_list[(feature_list >= left) & (feature_list < right)]

            if len(filtered_array)==0:
                if left !=0:
                    real_boundary.append(left)
                continue
            
            filtered_int = np.round(filtered_array,1)

            unique_elements, counts = np.unique(filtered_int, return_counts=True)            
            min_freq_bin = unique_elements[np.argmin(counts)]
            
            filtered_array = feature_list[(feature_list >= min_freq_bin-0.05) & (feature_list < (min_freq_bin+0.05))]            
            midian_value = self.find_min_frequency_value(filtered_array)

            if midian_value == 0:
                continue
            real_boundary.append(midian_value)
        return real_boundary
    
    def multi_fit(self, data):
        boundary_dict = dict()
        #data가 split_arrays임
        array_ = np.array(data)
        arrayT = array_.T
        for idx,feature_data in tqdm(enumerate(arrayT),total = len(arrayT)):
            if idx in self.ignore_idx:
                continue
            boundary_dict[idx] = self.make_boundary(self.n_log,feature_data)
        self.boundary_dict = boundary_dict
    
    def multi_transform(self,data_list):
        array_ = np.array(data_list)
        arrayT = array_.T
        
        for idx,data in tqdm(enumerate(arrayT),total=23):
            if idx in self.ignore_idx:
                data = data.astype(int)
                if idx == 0:
                    signature_array = np.array([chr(x + 65).zfill(2) for x in data])
                    continue
                else:
                    result_array = np.array([chr(x + 65).zfill(2) for x in data])
                    
            else:
                bins_ = np.digitize(data,self.boundary_dict[idx],right=True).astype(int)
                result_array = np.array([chr(int(x) + 65).zfill(2) for x in bins_])

            signature_array = np.vstack([signature_array, result_array]) 
        
        quan_data = [''.join(col) for col in signature_array.T]
        
        return quan_data


    
def make_log_quan(train_raw, train_key, dataset_path, n_com,dp_log):
    train_attack = []
    if global_.attack == 1:
        for idx, key in enumerate(train_key):
            if key.split('+')[0].upper() != "BENIGN":
                train_attack.append(train_raw[idx])
    elif global_.attack == 0:
        for idx, key in enumerate(train_key):
            if key.split('+')[0].upper() == "BENIGN":
                train_attack.append(train_raw[idx])
    else: 
        train_attack = train_raw

    pattern_log = log_Pattering(ignore_idx=[0, 1, 2], n_log_ = n_com, n_jobs=1)
    pattern_log.multi_fit(train_attack)

    with open(f"./preprocessing/{dataset_path}/LOG/{dp_log}", 'wb') as f:
        pickle.dump(pattern_log, f)

