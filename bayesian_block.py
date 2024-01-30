import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import multiprocessing as mp
import global_
from concurrent.futures import ProcessPoolExecutor
from astropy.stats import bayesian_blocks


class Bayesian_Block:
    def __init__(self, ignore_idx=[0,1,2], p0 = 0.05, n_jobs = 1):
        self.n_jobs = n_jobs
        self.ignore_idx = ignore_idx
        self.p0 = p0
        self.boundary = dict()
    
    def multi_fit(self, data_):
        data_ = np.array(data_)
        for i in range(len(data_[0])):
            data = data_[:, i].flatten()
            if i in self.ignore_idx:
                self.boundary[i] = []
                continue
            dim_data_nonzero = data[data != 0]
            edges = bayesian_blocks(dim_data_nonzero, fitness='events', p0=self.p0)

            self.boundary[i] = edges
    
    def fit(self, data):
        self.data_count = len(data)
        np_data = np.array(data)
        print("Model Fitting...")
        
        manager = mp.Manager()
        boundary_dict = manager.dict()

        # ProcessPoolExecutor를 사용하여 프로세스 풀 관리 개선
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for i in range(len(data[0])):
                # 각 프로세스에 대한 작업을 예약하고 futures 리스트에 추가
                future = executor.submit(self.multi_fit, i, np_data[:, i].flatten())
                futures.append(future)

            # 모든 프로세스의 완료를 기다리고 결과 확인
            for future in futures:
                try:
                    # 각 작업의 결과를 확인
                    result = future.result()
                except Exception as e:
                    # 예외 처리: 프로세스에서 발생한 예외를 처리
                    print(f"프로세스 실행 중 오류 발생: {e}")

    def multi_transform(self, data, i=0, result_dict=False):
        print(f'[{i}] Transform Start')
        np_data = np.array(data)
        ret_data = np.empty_like(np_data, dtype='<U12')
        
        for idx in range(len(data[0])):
            if idx in self.ignore_idx:
                ret_data[:, idx] = np.array(
                    list(map(lambda x: chr(int(float(x)) + 65).zfill(2), np_data[:, idx].astype('<U12'))))
                continue

            tmp_data = np_data[:, idx].flatten()
            pred = np.searchsorted(self.boundary[idx], tmp_data)

            new_data = []
            
            for p_idx in range(len(pred)):
                x = pred[p_idx]
                new_data.append(f'{x // 26}{chr(x % 26 + 65)}')
            ret_data[:, idx] = new_data

        tmp = [''.join(ret) for ret in ret_data]
        
        if self.n_jobs == 1:
            return tmp
        else:
            result_dict[i] = tmp

    def transform(self, data):
        if self.n_jobs == 1:
            print("Single CPU")
            ret_data = self.multi_transform(data)
        else:
            num_processes = self.n_jobs
            print(f"[{num_processes}] CPU")
            chunk_size = (len(data) // num_processes) + 1
            if chunk_size == 0:
                ret_data = self.multi_transform(data)
                return ret_data

            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            manager = mp.Manager()
            result_dict = manager.dict()

            processes = []
            for i, chunk in enumerate(tqdm(chunks, desc='Transform Start')):
                process = mp.Process(target=self.multi_transform, args=(chunk, i, result_dict))
                processes.append(process)
                process.start()

            for process in tqdm(processes, desc='Transform Finish'):
                process.join()

            processed_chunks = [result_dict[i] for i in range(len(result_dict))]
            ret_data = np.vstack(processed_chunks)

        return ret_data

def make_Bayesian(train_raw, train_key, p0, dp, dataset_path):
    train_attack = []
    if global_.attack == 1:
        for idx, key in enumerate(train_key):
            if key.split('+')[0].upper() != "BENIGN":
                    train_attack.append(train_raw[idx])

    elif global_.attack == 2:
        for idx, key in enumerate(train_key):
                train_attack.append(train_raw[idx])
    
    elif global_.attack == 0:
        for idx, key in enumerate(train_key):
            if key.split('+')[0].upper() == "BENIGN":
                train_attack.append(train_raw[idx])

    print(len(train_attack))
    pattern_gmm = Bayesian_Block(ignore_idx=[0, 1, 2], p0=p0, n_jobs=1)
    pattern_gmm.multi_fit(train_attack)

    with open(f"./preprocessing/{dataset_path}/Bayesian/{dp}", 'wb') as f:
        pickle.dump(pattern_gmm, f)

