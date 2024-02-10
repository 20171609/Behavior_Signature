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
    def make_countlist(self,data,ranges):
        counts = [0] * (len(ranges) + 1)
        
        # 이진 탐색으로 각 데이터를 적절한 범위에 할당하여 카운트
        for value in data:
            # 이진 탐색으로 적절한 범위 찾기
            left, right = 0, len(ranges)
            while left < right:
                mid = (left + right) // 2
                if value < ranges[mid]:
                        right = mid
                else:
                    left = mid + 1
            # 적절한 범위에 카운트 추가
            counts[left] += 1
        
        return counts
    def entropy_f(self,count_list,total_count):
        probabilities = [count / total_count for count in count_list]
        entropy = -sum(p * np.log2(p) for p in probabilities if p != 0)
        
        n = len(count_list)
        return entropy/np.log2(n)
    
    def make_boundary(self,logN_,feature_list):
        logN_n= logN_
        
        #min_data = min(feature_list)
        max_data = max(feature_list)
        #make original_list
        
        Nlog_list = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
        
        min_entropy = 999999
        before_entropy = 0 
        
        data_len = len(feature_list)
        
        real_boundary = [(2)**(i) for i in range(1,1000)]
        
        for logN in Nlog_list:
            original_list = [0]
            tmp=1
            i=1
            
            while(tmp<=max_data):
                tmp = logN**i
                i+=1
                original_list.append(tmp)
            
            len_ = len(original_list)
            
            #make real_boundary
            
            tmp_boundary = [-1,0]
            for idx,scale in enumerate(original_list):
                
                left = scale
                                
                if idx==(len_-1):
                    right = max_data+1    
                else:
                    right = original_list[idx+1]
                
                filtered_array = feature_list[(feature_list > left) & (feature_list <= right)]

                if len(filtered_array)==0:
                    if left !=0:
                        tmp_boundary.append(right)
                    continue
                midian_value = self.find_min_frequency_value(filtered_array)

                tmp_boundary.append(midian_value)
                
            count_list = self.make_countlist(feature_list,tmp_boundary)
            entropy = self.entropy_f(count_list,data_len)
            
            if min_entropy >= entropy:
                real_boundary = tmp_boundary
                min_entropy = entropy
                self.n_log = logN
            elif before_entropy < entropy:
                break
            before_entropy = entropy
        
        return real_boundary
    
    def multi_fit(self, data):
        boundary_dict = dict()
        
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
                result_array = np.array([f'{x // 26}{chr(x % 26 + 65)}' for x in bins_])

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

