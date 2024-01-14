import pickle

import numpy as np
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import global_

class GMM_Pattering:
    def __init__(self, ignore_idx=[], random_seed=43, covariance_type='full', max_iter=2000, n_components=10,
                 reg_covar=1e-6, tol=1e-3, dup_col={}, root_n=False, n_jobs=1):
        self.dup_col = dup_col
        self.reg_covar = reg_covar
        self.tol = tol
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.ignore_idx = ignore_idx
        self.models = []
        self.confidence = 0
        self.random_seed = random_seed
        self.data_count = 0
        self.table = {}
        self.root_n = root_n
        self.sort_index = {}
        self.n_jobs = n_jobs

    def multi_fit(self, idx, data, model_dict, sort_idx_dict, table_dict):
        if idx in self.ignore_idx:
            pass
        else:
            tmp_vgm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter,
                                      random_state=self.random_seed,
                                      covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                      tol=self.tol)
            tmp_vgm.fit(data)
            model_dict[idx] = tmp_vgm
            if self.root_n:
                weights = model_dict[idx].weights_ * self.data_count
            else:
                weights = np.ones_like(model_dict[idx].weights_)
            if self.covariance_type == 'full':
                table_dict[idx] = {
                    i: {"mean": model_dict[idx].means_[i][0],
                        "std": model_dict[idx].covariances_[i][0][0] ** 0.5 / (weights[i] ** 0.5)}
                    for i
                    in range(self.n_components)}
            elif self.covariance_type == 'spherical':
                table_dict[idx] = {
                    i: {"mean": model_dict[idx].means_[i][0],
                        "std": model_dict[idx].covariances_[i] ** 0.5 / (weights[i] ** 0.5)} for i
                    in
                    range(self.n_components)}
            sort_idx_dict[idx] = {i: idx for idx, i in enumerate(model_dict[idx].means_.reshape(-1).argsort())}
            print(f"Finish [{idx}] GMM")

    def fit(self, data):
        self.data_count = len(data)
        np_data = np.array(data)
        print("Model Fitting...")
        if self.n_jobs == 1:
            for idx in tqdm(range(len(data[0]))):
                if idx in self.ignore_idx:
                    tmp_vgm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter,
                                              random_state=self.random_seed,
                                              covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                              tol=self.tol)
                    self.models.append(tmp_vgm)
                    continue
                elif idx in self.dup_col:
                    tmp_vgm = self.models[self.dup_col[idx]]
                else:
                    tmp_data = np_data[:, idx].reshape(-1, 1)
                    tmp_vgm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter,
                                              random_state=self.random_seed,
                                              covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                              tol=self.tol)
                    tmp_vgm.fit(tmp_data)
                self.models.append(tmp_vgm)
                if self.root_n:
                    weights = tmp_vgm.weights_ * self.data_count
                else:
                    weights = np.ones_like(tmp_vgm.weights_)
                if self.covariance_type == 'full':
                    self.table[idx] = {
                        i: {"mean": tmp_vgm.means_[i][0],
                            "std": tmp_vgm.covariances_[i][0][0] ** 0.5 / (weights[i] ** 0.5)}
                        for i
                        in range(self.n_components)}
                elif self.covariance_type == 'spherical':
                    self.table[idx] = {
                        i: {"mean": tmp_vgm.means_[i][0], "std": tmp_vgm.covariances_[i] ** 0.5 / (weights[i] ** 0.5)}
                        for i
                        in
                        range(self.n_components)}
                self.sort_index[idx] = {i: idx for idx, i in enumerate(tmp_vgm.means_.reshape(-1).argsort())}
        else:
            manager = mp.Manager()
            model_dict = manager.dict()
            sort_idx_dict = manager.dict()
            table_dict = manager.dict()

            # ProcessPoolExecutor를 사용하여 프로세스 풀 관리 개선
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i in range(len(data[0])):
                    # 각 프로세스에 대한 작업을 예약하고 futures 리스트에 추가
                    future = executor.submit(self.multi_fit, i, np_data[:, i].reshape(-1, 1), model_dict, sort_idx_dict, table_dict)
                    futures.append(future)

                # 모든 프로세스의 완료를 기다리고 결과 확인
                for future in futures:
                    try:
                        # 각 작업의 결과를 확인
                        result = future.result()
                    except Exception as e:
                        # 예외 처리: 프로세스에서 발생한 예외를 처리
                        print(f"프로세스 실행 중 오류 발생: {e}")

            # 결과 수집
            self.table = {key: table_dict[key] for key in table_dict}
            self.sort_index = {key: sort_idx_dict[key] for key in sort_idx_dict}
            self.models = {key: model_dict[key] for key in model_dict}

            # 결과 수집
            self.table = {key: table_dict[key] for key in table_dict}
            self.sort_index = {key: sort_idx_dict[key] for key in sort_idx_dict}
            self.models = {key: model_dict[key] for key in model_dict}

    def multi_transform(self, data, i=0, result_dict=False):
        print(f'[{i}] Transform Start')
        np_data = np.array(data)
        ret_data = np.empty_like(np_data, dtype='<U12')
        for idx in range(len(data[0])):
            if idx in self.ignore_idx:
                ret_data[:, idx] = np.array(
                    list(map(lambda x: chr(int(float(x)) + 65).zfill(2), np_data[:, idx].astype('<U12'))))
                continue
            tmp_data = np_data[:, idx].reshape(-1, 1)
            pred = self.models[idx].predict(tmp_data).astype('<U12')
            for p_idx in range(len(pred)):
                tmp_pred = int(pred[p_idx])
                tmp_mean = self.table[idx][tmp_pred]['mean']
                tmp_std = self.table[idx][tmp_pred]['std']
                tmp_pred = self.sort_index[idx][tmp_pred] + 65
                if tmp_mean - (self.confidence * tmp_std) > tmp_data[p_idx]:
                    pred[p_idx] = f'-{chr(tmp_pred)}'
                elif tmp_data[p_idx] > tmp_mean + (self.confidence * tmp_std):
                    pred[p_idx] = f'-{chr(tmp_pred + 1)}'
                else:
                    pred[p_idx] = chr(tmp_pred).zfill(2)
            ret_data[:, idx] = pred
        if self.n_jobs == 1:
            return ret_data
        else:
            result_dict[i] = ret_data

    def transform(self, data, confidence=2.58):
        self.confidence = confidence
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

    def show_table(self):
        print("[n]  Duration  avg/std*sqrt(n)    Packets    avg/std*sqrt(n)    Bytes   avg/std*sqrt(n)")
        for n in range(self.n_components-1):
            print(
                f"[{n}] {self.table[3][n]['mean']}/{self.table[3][n]['std']}   {self.table[5][n]['mean']}/{self.table[5][n]['std']}   {self.table[7][n]['mean']}/{self.table[7][n]['std']}")
        return self.table

    def update_table(self, data_count, gmm_model):
        self.models = gmm_model
        for idx in range(len(self.models)):
            if idx in self.ignore_idx:
                continue
            if self.root_n:
                weights = self.models[idx].weights_ * data_count
            else:
                weights = np.ones_like(self.models[idx].weights_)
            if self.covariance_type == 'full':
                self.table[idx] = {
                    i: {"mean": self.models[idx].means_[i][0],
                        "std": self.models[idx].covariances_[i][0][0] ** 0.5 / (weights[i] ** 0.5)} for i in
                    range(self.n_components)}
            elif self.covariance_type == 'spherical':
                self.table[idx] = {
                    i: {"mean": self.models[idx].means_[i][0],
                        "std": self.models[idx].covariances_[i] ** 0.5 / (weights[i] ** 0.5)} for i in
                    range(self.n_components)}
            self.sort_index[idx] = {i: j for j, i in enumerate(self.models[idx].means_.reshape(-1).argsort())}

    def multi_tokenize(self, data, i, result_dict):
        result_dict[i] = list(map(lambda x: ''.join(x), data))

    def tokenize(self, data):
        if self.n_jobs == 1:
            token = list(map(lambda x: ''.join(x), data))
        else:
            num_processes = self.n_jobs
            chunk_size = len(data) // num_processes
            if chunk_size == 0:
                token = list(map(lambda x: ''.join(x), data))
                return token
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            manager = mp.Manager()
            result_dict = manager.dict()

            processes = []
            for i, chunk in enumerate(chunks):
                process = mp.Process(target=self.multi_tokenize, args=(chunk, i, result_dict))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            token = []
            for i in range(len(result_dict)):
                token += result_dict[i]
        return token

    def fit_transform(self, data, confidence=2.58):
        self.fit(data)
        return self.transform(data, confidence)

    def fit_transform_tokenize(self, data, confidence=2.58):
        tmp_data = self.fit_transform(data, confidence)
        return self.tokenize(tmp_data)

    def transform_tokenize(self, data, confidence=2.58):
        tmp_data = self.transform(data, confidence)
        return self.tokenize(tmp_data)

def make_gmm(train_raw, train_key, n_components, dp, dataset_path):
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

    pattern_gmm = GMM_Pattering(ignore_idx=[0, 1, 2], random_seed=43, n_components=n_components,\
                                max_iter=4000, n_jobs=6)
    pattern_gmm.fit(train_attack)

    with open(f"./preprocessing/{dataset_path}/GMM/{dp}", 'wb') as f:
        pickle.dump(pattern_gmm, f)

if __name__ == '__main__':
    gmm_pattern = GMM_Pattering(ignore_idx=[0, 1, 2], n_components=3, random_seed=43, n_jobs=12)

    with open(r"C:\jupyter_project\fsec 2023\test_path\Thursday-15-02-2018_feature2.pkl", "rb") as f:
        data = pickle.load(f)
    gmm_pattern.fit(data[:20000])

    with open("test.pkl", "wb") as f:
        pickle.dump(gmm_pattern, f)

    # with open("test.pkl", "rb") as f:
    #     gmm_pattern = pickle.load(f)

    print(gmm_pattern.table)
    print(gmm_pattern.transform_tokenize(data[:50]))
